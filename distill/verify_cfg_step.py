"""Isolate the CFG path at the single-step level.

`verify_teacher.py` compares final images after 50 Euler steps, where CFG with w=4 computes
`4*v_pos - 3*v_neg` and so amplifies each forward's bf16 rounding by 3-4x, compounding over
steps. That makes it a poor discriminator between "amplified numerical noise" and "wrong
wiring". This script compares tensors for a SINGLE step against the reference pipeline's own
`cfg_guided_model_fn`, so nothing accumulates:

  1. our negative prompt embedding vs the pipeline unit's negative branch
  2. our conditional velocity vs pipe.model_fn with the pipeline's own embeddings
  3. our CFG-combined velocity vs pipe.cfg_guided_model_fn
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill.fov_net import Flux2FoveatedNet, circular_mask  # noqa: E402
from distill.train_dmd import DEFAULT_MODEL_PATHS, DEFAULT_PROMPT  # noqa: E402


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / a.norm().clamp(min=1e-12)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_lora_path", default=None)
    p.add_argument("--lr_downsample_factor", type=int, default=2)
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--t", type=float, default=0.8)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = p.parse_args()

    net = Flux2FoveatedNet(
        model_id_with_origin_paths=DEFAULT_MODEL_PATHS,
        tokenizer_path="black-forest-labs/FLUX.2-klein-base-4B:tokenizer/",
        lora_path=args.teacher_lora_path,
        device="cuda",
        height=args.height,
        width=args.width,
        lr_downsample_factor=args.lr_downsample_factor,
        use_gradient_checkpointing=False,
    )
    net.to("cuda")
    net.set_role("teacher")
    pipe = net.pipe
    f, h, w = args.lr_downsample_factor, net.latent_height, net.latent_width
    mask = circular_mask(h, w, (-0.3, 0.0), 0.25, net.device)

    # ---- 1. embeddings: ours vs the pipeline's own posi/nega units ----------------
    unit = net._text_unit_qwen3
    ref_posi = unit.process(pipe, args.prompt)
    ref_nega = unit.process(pipe, "")
    ours_posi = net.encode_prompt([args.prompt])
    ours_nega = net.encode_prompt([""])
    e1 = rel(ref_posi["prompt_embeds"], ours_posi["prompt_embeds"])
    e2 = rel(ref_nega["prompt_embeds"], ours_nega["prompt_embeds"])
    print(f"[1] prompt_embeds rel.err  positive={e1:.3e}  negative={e2:.3e}")
    print(f"    text_ids equal: posi={torch.equal(ref_posi['text_ids'], ours_posi['text_ids'])} "
          f"nega={torch.equal(ref_nega['text_ids'], ours_nega['text_ids'])}")

    # ---- shared state for a single step -------------------------------------------
    torch.manual_seed(0)
    L = net.token_count(mask)
    x = torch.randn(1, L, 128, device=net.device, dtype=net.torch_dtype)
    t = torch.tensor([args.t], dtype=net.noise_scheduler.t_precision)

    ids, res_mask, res_tl = type(pipe)._prepare_latent_image_ids_foveation(
        1, h, w, net.device, x.dtype, mask, lr_factor=f
    )
    if ids.ndim == 2:
        ids = ids.unsqueeze(0)

    def ref_model_fn(embeds, text_ids):
        return pipe.model_fn(
            dit=pipe.dit, latents=x, timestep=t.to(net.device) * 1000,
            embedded_guidance=4.0, prompt_embeds=embeds, text_ids=text_ids,
            image_ids=ids, resolution_mask=res_mask, resolution_mask_top_left=res_tl,
            lr_downsample_factor=f, use_gradient_checkpointing=False,
        )

    with torch.no_grad():
        v_ref_posi = ref_model_fn(ref_posi["prompt_embeds"], ref_posi["text_ids"])
        v_ref_nega = ref_model_fn(ref_nega["prompt_embeds"], ref_nega["text_ids"])
        v_ref_cfg = v_ref_nega + args.cfg_scale * (v_ref_posi - v_ref_nega)

        cond = net.make_condition(ours_posi, mask)
        neg = net.make_condition(ours_nega, mask)
        v_ours_posi = net(x, t, condition=cond, fwd_pred_type="flow")
        v_ours_nega = net(x, t, condition=neg, fwd_pred_type="flow")
        v_ours_cfg = v_ours_nega + args.cfg_scale * (v_ours_posi - v_ours_nega)

    print(f"[2] velocity rel.err  conditional={rel(v_ref_posi, v_ours_posi):.3e}  "
          f"unconditional={rel(v_ref_nega, v_ours_nega):.3e}")
    print(f"[3] CFG-combined (w={args.cfg_scale:g}) rel.err={rel(v_ref_cfg, v_ours_cfg):.3e}")

    # how much does CFG amplify a fixed perturbation? explains image-space MAE growth
    amp = (v_ref_cfg.float().norm() / v_ref_posi.float().norm()).item()
    print(f"    ||v_cfg||/||v_cond|| = {amp:.3f}   "
          f"cond-vs-uncond rel.diff = {rel(v_ref_posi, v_ref_nega):.3e}")

    # bf16 noise floor: rerun the SAME call and see how much it moves on its own
    with torch.no_grad():
        v_again = net(x, t, condition=cond, fwd_pred_type="flow")
    print(f"[4] bf16 nondeterminism floor (same call twice) = {rel(v_ours_posi, v_again):.3e}")

    tol = 5e-3
    ok = (max(e1, e2) < 1e-5
          and rel(v_ref_posi, v_ours_posi) < tol
          and rel(v_ref_nega, v_ours_nega) < tol
          and rel(v_ref_cfg, v_ours_cfg) < tol)
    print("PASS -- CFG path matches the reference at single-step level" if ok else
          "FAIL -- CFG path diverges beyond the numerical floor")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
