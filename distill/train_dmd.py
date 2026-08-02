"""Foveated DMD/DMD2 distillation of FLUX.2-klein on reduced mixed-resolution tokens.

    bash distill/run_dmd.sh                 # full run  (edit GPU_IDS first!)
    GPU_IDS=3,4,5,6 bash distill/run_dmd_smoke.sh    # correctness checks + a few steps

Design notes
------------
* One shared frozen DiT (the foveated SFT checkpoint, LoRA fused in) serves as the
  *teacher*; two PEFT adapters on top are the *student* and the *critic* (DMD fake score).
  One 6.1 GB transformer instead of three.
* Distributed training is accelerate for process/dataloader/AMP management, but gradients
  are all-reduced manually over the active adapter's parameters. This is deliberate: DDP's
  bucketing assumes a stable autograd graph, and DMD alternates between two disjoint
  subgraphs (student vs critic) every iteration. Manual all-reduce over ~4 M LoRA params is
  cheap and side-steps that entirely.
* The foveation mask is redrawn every iteration exactly as in the SFT loss, so the token
  count L varies (~1200-2250 at 1024px with f=2). One mask per iteration, shared across the
  batch, since different masks give different L and cannot be stacked.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill.data import AestheticFoveatedDataset, collate  # noqa: E402
from distill.dmd import FoveatedDMD  # noqa: E402
from distill.fov_net import (  # noqa: E402
    VALIDATION_POSITIONS,
    VALIDATION_RADIUS,
    Flux2FoveatedNet,
    circular_mask,
)

DEFAULT_MODEL_PATHS = (
    "black-forest-labs/FLUX.2-klein-base-4B:text_encoder/*.safetensors,"
    "black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,"
    "black-forest-labs/FLUX.2-klein-base-4B:vae/diffusion_pytorch_model.safetensors"
)
DEFAULT_PROMPT = (
    "Golden retriever on a grassy meadow. Thick sun-kissed fur, clean and glossy, subtle "
    "lighter ruff around the neck. Alert amber eyes, black moist nose, relaxed open mouth "
    "with pink tongue. Perky ears tilted forward, sturdy athletic stance, tail held in a "
    "gentle arc. Warm late-afternoon backdrop, rolling wild grass bending in the breeze, "
    "blurred tree line and distant hills under a pastel sky."
)


def parse_args():
    p = argparse.ArgumentParser("Foveated DMD distillation")
    # data / model
    p.add_argument("--dataset_base_path", default="/local/brian/Aesthetic-Train-V2/")
    p.add_argument("--dataset_metadata_path", default="/local/brian/Aesthetic-Train-V2/metadata_train.csv")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--model_id_with_origin_paths", default=DEFAULT_MODEL_PATHS)
    p.add_argument("--tokenizer_path", default="black-forest-labs/FLUX.2-klein-base-4B:tokenizer/")
    p.add_argument("--teacher_lora_path", default=None,
                   help="Foveated SFT LoRA, fused into the base to form the teacher.")
    p.add_argument("--teacher_lora_alpha", type=float, default=1.0)
    p.add_argument("--student_lora_init", default=None,
                   help="Optional .safetensors to warm-start the student adapter.")
    # shape / foveation
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--lr_downsample_factor", type=int, default=2, choices=[2, 4])
    p.add_argument("--embedded_guidance", type=float, default=4.0,
                   help="Inert for FLUX.2-klein-base-4B (no guidance embedder in the "
                        "checkpoint); kept for models that do have one.")
    # lora
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_target_modules", default="")
    # dmd
    p.add_argument("--student_steps", type=int, default=4)
    p.add_argument("--student_t_mode", default="flux2", choices=["uniform", "flux2"],
                   help="flux2 (default): the student is sampled at the SAME timesteps the "
                        "pretrained scheduler produces for --student_steps, so the distilled "
                        "LoRA drops straight into Flux2FoveatedImagePipeline at that step "
                        "count and stays in the noise regime the base model is calibrated "
                        "for. uniform is FastGen's get_t_list default.")
    p.add_argument("--student_mu", type=float, default=None,
                   help="Override the shift (mu) of the student few-step schedule; "
                        "shift=exp(mu). Default None = the pipeline's own value for "
                        "--student_steps (mu=2.29 / shift=9.89 at 4 steps). A STUDENT "
                        "hyperparameter worth sweeping upward; a custom value gives up "
                        "drop-in inference (pipeline must be handed matching sigmas).")
    p.add_argument("--student_sample_type", default="sde", choices=["sde", "ode"])
    p.add_argument("--critic_update_freq", type=int, default=5,
                   help="Student is updated once every N iterations; critic gets the rest.")
    p.add_argument("--critic_warmup_steps", type=int, default=100,
                   help="Critic-only iterations before the first student update. With a "
                        "zero-init critic the VSD gradient is exactly zero, so the student "
                        "would waste its first updates.")
    p.add_argument("--time_dist", default="shifted",
                   choices=["uniform", "logitnormal", "shifted"],
                   help="Distribution for the DMD comparison timestep. 'shifted' with "
                        "--time_shift exp(0.8) reproduces the FLUX.2 training density used by "
                        "the SFT runs; 'uniform' is FastGen's default.")
    p.add_argument("--time_shift", type=float, default=None,
                   help="shift for --time_dist shifted; default exp(0.8)=2.2255, i.e. FLUX.2's mu=0.8.")
    p.add_argument("--min_t", type=float, default=0.001)
    p.add_argument("--max_t", type=float, default=0.999)
    p.add_argument("--student_cfg_scale", type=float, default=1.0,
                   help="CFG the STUDENT is sampled with. Set equal to --teacher_cfg_scale to "
                        "isolate step distillation (guidance fixed on both sides, student "
                        "does 2 forwards/step). Leave at 1.0 to additionally distil guidance "
                        "away -- two entangled objectives, much harder to debug.")
    p.add_argument("--teacher_cfg_scale", type=float, default=4.0,
                   help="True CFG scale for the teacher. FLUX.2-klein-base has NO guidance "
                        "embedder, so this is the only guidance available; 1.0 distils an "
                        "unguided model. Costs a second teacher forward on generator steps.")
    # optim
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--student_lr", type=float, default=5e-6)
    p.add_argument("--critic_lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9,
                   help="DMD2 upstream uses 0.0, which makes each Adam step ~lr*sign(g) -- "
                        "fine with their large batches, but it does not average out the "
                        "gradient noise of a small effective batch. 0.9 by default here.")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_train_steps", type=int, default=5000)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # io / logging
    p.add_argument("--output_path", default="./outputs/dmd_fov")
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--validation_steps", type=int, default=500)
    p.add_argument("--validation_prompts", nargs="+", default=[DEFAULT_PROMPT])
    p.add_argument("--teacher_val_steps", type=int, default=50)
    p.add_argument("--skip_initial_validation", action="store_true")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="flux2-foveated-dmd")
    p.add_argument("--wandb_run_name", default=None)
    # smoke test
    p.add_argument("--smoke_test", action="store_true",
                   help="Run correctness assertions, a few train steps and one validation, then exit.")
    return p.parse_args()


# --------------------------------------------------------------------------- ddp
def all_reduce_grads(params, world_size: int):
    if world_size == 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size


def save_lora(net: Flux2FoveatedNet, role: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = net.lora_state_dict(role)
    # rename the adapter infix to `default` so DiffSynth's loaders can consume the file
    sd = {k.replace(f".{role}.", ".default."): v.contiguous().to(torch.float32) for k, v in sd.items()}
    save_file(sd, path)


# -------------------------------------------------------------------- validation
@torch.no_grad()
def run_validation(args, model: FoveatedDMD, accelerator: Accelerator, step: int, wandb_run):
    """Two wandb panels per validation.

    Group 1 `validation/center_quality` -- centred mask, three ways from IDENTICAL noise:
        teacher @ --teacher_val_steps (the quality ceiling),
        teacher @ --student_steps     (the naive few-step baseline DMD has to beat),
        student @ --student_steps     (the distilled model).
    Group 2 `validation/student_masks` -- the student at the four mask positions, i.e. the
        gaze-agnostic behaviour the SFT script visualises.

    Teacher sampling is CFG-guided (this DiT has no guidance embedding), students are not.
    """
    if not accelerator.is_main_process:
        return
    net = model.net
    h, w = net.latent_height, net.latent_width
    from PIL import Image as PILImage

    def _noise(L):
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        return torch.randn(1, L, 128, generator=g).to(net.device, net.torch_dtype)

    def _sde_gen():
        # fresh, identically-seeded generator per sample so student panels are comparable
        # across validations; otherwise SDE re-noising alone moves the image by ~10/255.
        return torch.Generator(device="cpu").manual_seed(args.seed + 1)

    import numpy as _np

    def _mae(a, b):
        return float(_np.abs(_np.asarray(a, dtype=_np.float32)
                             - _np.asarray(b, dtype=_np.float32)).mean())

    center_imgs, center_caps = [], []
    mask_imgs, student_imgs, student_caps = [], [], []
    metrics = {}

    for prompt in args.validation_prompts:
        text = net.encode_prompt([prompt])
        neg_text = net.encode_prompt([""])
        tag = prompt[:32].strip()

        # ---- group 1: centred mask, teacher-50 / teacher-4 / student-4 --------------
        mask_c = circular_mask(h, w, (0.0, 0.0), VALIDATION_RADIUS, net.device)
        cond_c = net.make_condition(text, mask_c)
        neg_c = net.make_condition(neg_text, mask_c)
        Lc = net.token_count(mask_c)
        noise_c = _noise(Lc)

        z = net.sample_teacher(noise_c, cond_c, num_steps=args.teacher_val_steps,
                               cfg_scale=args.teacher_cfg_scale, neg_condition=neg_c)
        center_imgs.append(net.decode_fov_latents(z, mask_c)[0])
        center_caps.append(f"teacher {args.teacher_val_steps}-step cfg{args.teacher_cfg_scale:g} "
                           f"| L={Lc} | {tag}")

        z = net.sample_teacher(noise_c, cond_c, num_steps=args.student_steps,
                               cfg_scale=args.teacher_cfg_scale, neg_condition=neg_c)
        center_imgs.append(net.decode_fov_latents(z, mask_c)[0])
        center_caps.append(f"teacher {args.student_steps}-step cfg{args.teacher_cfg_scale:g} "
                           f"| L={Lc} | {tag}")  # == the no-cfg one when cfg==1

        # Same-compute, same-guidance reference: the teacher few-step WITHOUT cfg is exactly
        # what the student is at init (verified: 34.91 vs 35.00 MAE). Plotting it shows that
        # the student starts where it should, and that the student->cfg-teacher gap is the
        # actual distillation task (absorb CFG + few steps into one unguided pass).
        # init reference: teacher Euler on the STUDENT's own grid at the STUDENT's guidance.
        # Because the `ode` branch is exactly the Euler update, the student reproduces this at
        # initialisation, so `progress` starts at 0% by construction.
        z = net.sample_teacher(noise_c, cond_c, cfg_scale=args.student_cfg_scale,
                               neg_condition=neg_c,
                               sigmas=model.t_list[:-1].to(net.device))
        center_imgs.append(net.decode_fov_latents(z, mask_c)[0])
        center_caps.append(f"teacher {args.student_steps}-step cfg{args.student_cfg_scale:g} "
                           f"student-grid (init ref) | L={Lc} | {tag}")

        z = net.sample_student(noise_c, cond_c, model.t_list,
                               sample_type=args.student_sample_type, generator=_sde_gen(),
                               cfg_scale=args.student_cfg_scale, neg_condition=neg_c)
        center_imgs.append(net.decode_fov_latents(z, mask_c)[0])
        center_caps.append(f"student {args.student_steps}-step {args.student_sample_type} "
                           f"(distilled) | L={Lc} | {tag}")

        z = net.sample_student(noise_c, cond_c, model.t_list, sample_type="ode",
                               cfg_scale=args.student_cfg_scale, neg_condition=neg_c)
        center_imgs.append(net.decode_fov_latents(z, mask_c)[0])
        center_caps.append(f"student {args.student_steps}-step ode | L={Lc} | {tag}")

        # The actual research claim, as a scalar: the student's few-step output should move
        # TOWARD the 50-step teacher and past the naive few-step teacher baseline.
        t50, t4, t4n, stu, stu_ode = center_imgs[-5:]
        metrics.setdefault("val/student_vs_teacher50_mae", []).append(_mae(stu, t50))
        metrics.setdefault("val/student_ode_vs_teacher50_mae", []).append(_mae(stu_ode, t50))
        metrics.setdefault("val/teacher4_cfg_vs_teacher50_mae", []).append(_mae(t4, t50))
        metrics.setdefault("val/teacher4_nocfg_vs_teacher50_mae", []).append(_mae(t4n, t50))

        # ---- group 2: student at the four mask positions ----------------------------
        for pos_name, center in VALIDATION_POSITIONS:
            mask = circular_mask(h, w, center, VALIDATION_RADIUS, net.device)
            cond = net.make_condition(text, mask)
            # The negative condition MUST carry this loop's mask: the mask determines the
            # sequence length L and the RoPE ids, so reusing another mask's condition makes
            # the ids disagree with the latents ("resolution mask shape N mismatch").
            neg = net.make_condition(neg_text, mask) if neg_text is not None else None
            L = net.token_count(mask)
            z = net.sample_student(_noise(L), cond, model.t_list,
                                   sample_type=args.student_sample_type, generator=_sde_gen(),
                                   cfg_scale=args.student_cfg_scale, neg_condition=neg)
            student_imgs.append(net.decode_fov_latents(z, mask)[0])
            student_caps.append(f"student {args.student_steps}-step [{pos_name}] | L={L} | {tag}")
            if step == 0:
                m = (mask.float().cpu().numpy() * 255).astype("uint8")
                mask_imgs.append((PILImage.fromarray(m), f"mask_{pos_name}"))
    if step == 0 and args.validation_prompts:
        m = (mask_c.float().cpu().numpy() * 255).astype("uint8")
        mask_imgs.append((PILImage.fromarray(m), "mask_center"))

    agg = {k: sum(v) / len(v) for k, v in metrics.items()}
    if agg:
        init_ref = agg["val/teacher4_nocfg_vs_teacher50_mae"]
        target = agg["val/teacher4_cfg_vs_teacher50_mae"]
        stu = agg["val/student_vs_teacher50_mae"]
        stu_o = agg["val/student_ode_vs_teacher50_mae"]
        # progress in [0, 1]: 0 = still at the unguided-teacher starting point,
        # 1 = matched the CFG teacher at the same step count.
        # The student is ALWAYS unguided, so the honest same-compute baseline is the
        # unguided few-step teacher; the ideal is 0 (matching the many-step reference).
        # Using the *guided* few-step teacher as the denominator (as before) mixes the
        # guidance axis back into the progress number and reads negative at init.
        # guard: if the reference and the init-ref coincide (e.g. teacher_val_steps ==
        # student_steps in a smoke test) the denominator is ~0 and progress is meaningless.
        prog = float("nan") if init_ref < 1e-3 else 1.0 - min(stu, stu_o) / init_ref
        extra = "" if abs(args.teacher_cfg_scale - 1.0) < 1e-9 else \
            f" | (cfg{args.teacher_cfg_scale:g} few-step teacher {target:.3f})"
        print(f"[val] step {step}: student sde {stu:.3f} ode {stu_o:.3f} | "
              f"init-ref (teacher{args.student_steps} cfg{args.student_cfg_scale:g}) "
              f"{init_ref:.3f} "
              f"-> ref (teacher{args.teacher_val_steps}) 0 | "
              f"progress {prog*100:+.1f}%{extra}"
              if prog == prog else
              f"[val] step {step}: student sde {stu:.3f} ode {stu_o:.3f} | init-ref "
              f"{init_ref:.3f} degenerate (== reference), progress undefined{extra}")

    if wandb_run is not None:
        import wandb

        wandb_run.log(agg, step=step)
        wandb_run.log(
            {
                "validation/center_quality": [
                    wandb.Image(i, caption=c) for i, c in zip(center_imgs, center_caps)
                ],
                "validation/student_masks": [
                    wandb.Image(i, caption=c) for i, c in zip(student_imgs, student_caps)
                ],
            },
            step=step,
        )
        if mask_imgs:
            wandb_run.log(
                {"validation/foveation_masks": [wandb.Image(i, caption=c) for i, c in mask_imgs]},
                step=step,
            )

    out_dir = os.path.join(args.output_path, "validation", f"step-{step}")
    os.makedirs(out_dir, exist_ok=True)

    def _slug(s):
        return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)[:90]

    for i, (img, cap) in enumerate(zip(center_imgs, center_caps)):
        img.save(os.path.join(out_dir, f"center_{i}_{_slug(cap)}.png"))
    for i, (img, cap) in enumerate(zip(student_imgs, student_caps)):
        img.save(os.path.join(out_dir, f"student_{i}_{_slug(cap)}.png"))
    print(f"[val] step {step}: wrote {len(center_imgs)} center + {len(student_imgs)} "
          f"student-mask images to {out_dir}")


# ------------------------------------------------------------------ smoke checks
@torch.no_grad()
def correctness_checks(net: Flux2FoveatedNet, model: FoveatedDMD, accelerator: Accelerator):
    """Assertions on the time convention, the schedule algebra and the foveated shapes.

    These are the failure modes that would otherwise train silently on garbage.
    """
    ok = lambda name, cond: print(f"  [{'PASS' if cond else 'FAIL'}] {name}") or cond
    results = []
    dev, dt = net.device, net.torch_dtype
    sched = net.noise_scheduler

    # 1. rf schedule matches DiffSynth's FLUX.2 add_noise / training_target exactly
    from diffsynth.diffusion.flow_match import FlowMatchScheduler

    ds = FlowMatchScheduler("FLUX.2")
    ds.set_timesteps(1000, training=True)
    x0 = torch.randn(1, 8, 4, dtype=torch.float32)
    eps = torch.randn_like(x0)
    idx = 300
    sigma = ds.sigmas[idx]
    t = torch.tensor([sigma.item()], dtype=sched.t_precision)
    a = ds.add_noise(x0, eps, ds.timesteps[idx])
    b = sched.forward_process(x0, eps, t)
    results.append(ok(f"forward_process == diffsynth.add_noise (sigma={sigma:.4f})",
                      torch.allclose(a, b, atol=1e-5)))
    v = ds.training_target(x0, eps, ds.timesteps[idx])
    results.append(ok("flow target == eps - x0", torch.allclose(v, eps - x0)))
    results.append(ok("flow_to_x0 inverts the forward process",
                      torch.allclose(sched.flow_to_x0(b, v, t), x0, atol=1e-4)))

    # 2. token bookkeeping: L from the mask matches what _downsample_latents produces
    mask = net.sample_mask()
    L = net.token_count(mask)
    hr = torch.randn(1, net.full_seq_len, 128, device=dev, dtype=dt)
    lo = torch.randn(
        1, net.full_seq_len // (net.lr_factor**2), 128, device=dev, dtype=dt
    )
    merged = net.pipe._downsample_latents(
        hr, net.latent_height, net.latent_width, foveation_mask=mask,
        token_downsampler=None, latents_downsampled=lo, lr_factor=net.lr_factor,
    )
    results.append(ok(f"token_count() == merged length ({L} vs {merged.shape[1]})",
                      L == merged.shape[1]))
    results.append(ok(f"L < full_seq_len ({L} < {net.full_seq_len})", L < net.full_seq_len))

    # 3. the DiT accepts the reduced sequence and returns the same length
    text = net.encode_prompt(["a photo of a cat"])
    cond = net.make_condition(text, mask)
    tt = torch.tensor([0.5], dtype=sched.t_precision)
    z = torch.randn(1, L, 128, device=dev, dtype=dt)
    with net.role("teacher"):
        v_out = net(z, tt, condition=cond, fwd_pred_type="flow")
    results.append(ok(f"DiT output shape preserved {tuple(v_out.shape)}", v_out.shape == z.shape))
    results.append(ok("DiT output finite", torch.isfinite(v_out).all().item()))

    # 4. the three roles are genuinely different networks once adapters are perturbed
    for n, p in net.pipe.dit.named_parameters():
        if "lora_B.student" in n:
            torch.nn.init.normal_(p, std=0.01)
    with net.role("student"):
        v_s = net(z, tt, condition=cond, fwd_pred_type="flow")
    with net.role("critic"):
        v_c = net(z, tt, condition=cond, fwd_pred_type="flow")
    with net.role("teacher"):
        v_t = net(z, tt, condition=cond, fwd_pred_type="flow")
    results.append(ok("student != teacher after perturbing student LoRA",
                      not torch.allclose(v_s, v_t)))
    results.append(ok("critic == teacher while its LoRA is still zero-init",
                      torch.allclose(v_c, v_t)))
    for n, p in net.pipe.dit.named_parameters():
        if "lora_B.student" in n:
            torch.nn.init.zeros_(p)

    # 5. x0 prediction of the teacher at high noise should be far tamer than the input
    with net.role("teacher"):
        x0_hat = net(z, torch.tensor([0.999], dtype=sched.t_precision), condition=cond,
                     fwd_pred_type="x0")
    results.append(ok(f"teacher x0 std {x0_hat.float().std():.3f} < input std "
                      f"{z.float().std():.3f}", x0_hat.float().std() < z.float().std()))

    # 6. student timestep list is descending and terminates at 0
    tl = model.t_list
    results.append(ok(f"student t_list {[round(float(x),4) for x in tl]} descending to 0",
                      bool((tl[:-1] > tl[1:]).all()) and float(tl[-1]) == 0.0
                      and float(tl[0]) <= net.t_max))
    return all(results)


# --------------------------------------------------------------------- main loop
def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    set_seed(args.seed + accelerator.process_index)
    world = accelerator.num_processes
    is_main = accelerator.is_main_process

    if is_main:
        os.makedirs(args.output_path, exist_ok=True)
        print(f"[setup] world_size={world} device={accelerator.device}")

    net = Flux2FoveatedNet(
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        lora_path=args.teacher_lora_path,
        lora_alpha=args.teacher_lora_alpha,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
        device=accelerator.device,
        height=args.height,
        width=args.width,
        lr_downsample_factor=args.lr_downsample_factor,
        embedded_guidance=args.embedded_guidance,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        max_t=args.max_t,
    )
    net.to(accelerator.device)

    if args.student_lora_init:
        from diffsynth.core import load_state_dict as _lsd

        sd = _lsd(args.student_lora_init)
        sd = {k.replace(".default.", ".student."): v for k, v in sd.items()}
        missing = net.pipe.dit.load_state_dict(sd, strict=False)
        if is_main:
            print(f"[setup] warm-started student from {args.student_lora_init} "
                  f"({len(sd)} tensors, {len(missing.unexpected_keys)} unexpected)")

    model = FoveatedDMD(
        net,
        student_steps=args.student_steps,
        student_t_mode=args.student_t_mode,
        student_mu=args.student_mu,
        student_sample_type=args.student_sample_type,
        critic_update_freq=args.critic_update_freq,
        critic_warmup_steps=args.critic_warmup_steps,
        time_dist=args.time_dist,
        **({"time_shift": args.time_shift} if args.time_shift is not None else {}),
        min_t=args.min_t,
        max_t=args.max_t,
        teacher_cfg_scale=args.teacher_cfg_scale,
        student_cfg_scale=args.student_cfg_scale,
    )

    student_params = net.lora_parameters("student")
    critic_params = net.lora_parameters("critic")
    assert student_params and critic_params, "LoRA adapters were not injected"
    betas = (args.adam_beta1, 0.999)
    opt_s = torch.optim.AdamW(student_params, lr=args.student_lr,
                              weight_decay=args.weight_decay, betas=betas)
    opt_c = torch.optim.AdamW(critic_params, lr=args.critic_lr,
                              weight_decay=args.weight_decay, betas=betas)
    if is_main:
        n_s = sum(p.numel() for p in student_params)
        eff = args.batch_size * accelerator.num_processes * max(1, args.gradient_accumulation_steps)
        print(f"[setup] trainable: student {n_s/1e6:.2f}M, critic {n_s/1e6:.2f}M | "
              f"student t_list={[round(float(x),4) for x in model.t_list]}")
        print(f"[setup] effective batch {eff} "
              f"(bs{args.batch_size} x {accelerator.num_processes} gpu x accum"
              f"{max(1, args.gradient_accumulation_steps)}), betas=({args.adam_beta1}, 0.999)")

    wandb_run = None
    if args.use_wandb and is_main:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                               config=vars(args), dir=args.output_path)

    if args.smoke_test:
        if is_main:
            print("[smoke] correctness checks")
        passed = correctness_checks(net, model, accelerator)
        if not passed:
            raise SystemExit("correctness checks FAILED")
        accelerator.wait_for_everyone()

    dataset = AestheticFoveatedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=collate, drop_last=True, pin_memory=True,
    )
    dataloader = accelerator.prepare(dataloader)

    if not args.skip_initial_validation:
        run_validation(args, model, accelerator, 0, wandb_run)
    accelerator.wait_for_everyone()

    # the negative ("") prompt embedding is prompt-independent, so encode it once
    neg_text = (net.encode_prompt([""])
                if (args.teacher_cfg_scale != 1.0 or args.student_cfg_scale != 1.0) else None)
    if is_main:
        print(f"[setup] teacher CFG scale {args.teacher_cfg_scale} "
              f"(guidance embedder present: {net.has_guidance_embed})")

    step, t0 = 0, time.time()
    running = {}
    done = False
    accum = max(1, args.gradient_accumulation_steps)
    micro = 0
    generator_iter = model.is_generator_iter(0)
    opt, params = (opt_s, student_params) if generator_iter else (opt_c, critic_params)
    opt.zero_grad(set_to_none=True)

    while not done:
        for batch in dataloader:
            if step >= args.max_train_steps:
                done = True
                break

            image = batch["image"].to(accelerator.device, net.torch_dtype)
            mask = net.sample_mask()  # fresh random mask per micro-batch, as in SFT
            text = net.encode_prompt(batch["prompt"])
            cond = net.make_condition(text, mask)
            real = net.encode_to_fov_latents(image, mask)
            # unconditional branch for the teacher's CFG (student/critic never use it)
            neg_cond = None
            if args.teacher_cfg_scale != 1.0 or args.student_cfg_scale != 1.0:
                neg_cond = net.make_condition(neg_text, mask)

            if generator_iter:
                loss, logs = model.generator_loss(real, cond, neg_condition=neg_cond)
            else:
                loss, logs = model.critic_loss(real, cond, neg_condition=neg_cond)

            # The adapter that built the graph must still be active: activation
            # checkpointing replays the blocks during backward.
            expected_role = "student" if generator_iter else "critic"
            assert net._role == expected_role, (
                f"role is {net._role!r} but the graph was built under {expected_role!r}; "
                "backward would recompute with the wrong adapter"
            )
            # Gradient accumulation is done by hand (not `accelerator.accumulate`) because the
            # optimizer, the trainable parameter set and the autograd subgraph all switch
            # between micro-batches groups; each micro-batch draws its own mask and t, so
            # accumulating genuinely averages over the mask/timestep distribution.
            accelerator.backward(loss / accum)

            logs["diag/L"] = torch.tensor(float(real.shape[1]))
            for k, v in logs.items():
                running.setdefault(k, []).append(float(v))

            micro += 1
            if micro < accum:
                continue
            micro = 0

            all_reduce_grads(params, world)
            gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            opt.step()
            running.setdefault(
                f"grad_norm/{'student' if generator_iter else 'critic'}", []
            ).append(float(gnorm.detach()))

            step += 1
            # choose the role for the NEXT optimizer step and reset its grads
            generator_iter = model.is_generator_iter(step)
            opt, params = (opt_s, student_params) if generator_iter else (opt_c, critic_params)
            opt.zero_grad(set_to_none=True)
            if step % args.log_steps == 0:
                means = {k: sum(v) / len(v) for k, v in running.items()}
                running = {}
                means["perf/steps_per_sec"] = args.log_steps / (time.time() - t0)
                t0 = time.time()
                if is_main:
                    if wandb_run is not None:
                        wandb_run.log(means, step=step)
                    head = " ".join(
                        f"{k.split('/')[-1]}={v:.4g}"
                        for k, v in means.items()
                        if k.startswith(("loss/", "diag/L", "perf/"))
                    )
                    print(f"[step {step}] {head}")

            if args.validation_steps and step % args.validation_steps == 0:
                run_validation(args, model, accelerator, step, wandb_run)
                accelerator.wait_for_everyone()

            if args.save_steps and step % args.save_steps == 0 and is_main:
                save_lora(net, "student", os.path.join(args.output_path, f"student-step-{step}.safetensors"))
                save_lora(net, "critic", os.path.join(args.output_path, f"critic-step-{step}.safetensors"))
                print(f"[save] step {step}")

            if args.smoke_test and step >= model.critic_warmup_steps + 2 * args.critic_update_freq:
                done = True
                break

    if args.smoke_test:
        run_validation(args, model, accelerator, step, wandb_run)
    if is_main:
        save_lora(net, "student", os.path.join(args.output_path, "student-final.safetensors"))
        save_lora(net, "critic", os.path.join(args.output_path, "critic-final.safetensors"))
        print("[done]")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
