#!/usr/bin/env python
"""ToMe-2x2 and RALU-foveated rows for the supplementary baseline figures.

`paper_figures/supp/baseline_{1,2,3,101}.png` tile
`results/eccv2026_final_v2/baseline_comparison_randomized_{1,2,3}/stacked/<id>.png`, each of
which is a horizontal stack of `high_res | naive | ours` written by
`inference.py --experiment user_study`. This script reproduces, for every one of those
tiles, the *same prompt* and the *same foveation mask* under two additional baselines, so
the extra columns can be stacked in without regenerating anything that already exists.

Prompt sets (folder index == CSV row index, verified 100/100 for all three, because
`run_user_study` skips its `random.sample` when `num_prompts == len(csv)`):

    set 1 -> prompts/claude_prompts.csv
    set 2 -> prompts/claude_prompts_extended.csv
    set 3 -> prompts/claude_prompts_extended_v2.csv

Mask reproduction
-----------------
`run_user_study` draws `center = (uniform(-0.3, 0.3), uniform(-0.3, 0.3))` per prompt from
the `random` module seeded once with `--seed`, with `r` fixed at 0.33, shape circular,
`lr_downsample_factor` 2 (inference.py:1005-1015). The draws are the first consumers of the
RNG, so replaying the seed reproduces the centers exactly. All three sets used seed 42 and
100 prompts, so they share one center sequence. `--verify_centers` checks the replay against
the saved `fixation_point.png` files before any model is loaded.

Model
-----
`run_user_study` clears LoRA before the high_res and naive passes and only loads the
foveation LoRA for "ours" (inference.py:1001, 1076). The two columns these new baselines sit
beside are therefore *base* FLUX.2-klein, so `--lora_checkpoint` defaults to None here —
unlike `run_tome.sh` / `ralu.sh`, which default to the no-foveation LoRA.

Compute matching
----------------
Both arms are matched to ours at 50 steps on the same mask:
  * ToMe  — merges the periphery into 2x2 blocks, so it runs our exact token budget every
            step by construction. `--merge_mlp true` is the headline config (RUN.md).
  * RALU  — `--match_compute` solves the total step count so RALU's token-steps equal
            `50 * (M_down + 3m)` with RALU's stage proportions held fixed, per prompt (the
            mask, and hence m, moves with the fixation point). `--z_max 6.5` keeps NT-DM at
            the re-noising strength the authors published (RALU_FOVEATED.md).

Usage
-----
    ./run_new_baselines.sh              # the launcher; see it for env knobs
    python new_baselines_inference.py --arm ralu --plan       # schedules only, no GPU
    python new_baselines_inference.py --verify_centers        # mask replay check, no GPU
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, os.path.join(EVAL_DIR, "importance-token-merge"))
sys.path.insert(0, os.path.join(EVAL_DIR, "RALU"))

PROMPT_SETS = {
    1: "prompts/claude_prompts.csv",
    2: "prompts/claude_prompts_extended.csv",
    3: "prompts/claude_prompts_extended_v2.csv",
}

# Fixed by run_user_study; not knobs.
USER_STUDY_RADIUS = 0.33
USER_STUDY_SHAPE = "circular"


def _str_to_bool(s):
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ("1", "true", "t", "yes", "y")


def user_study_centers(seed, n):
    """Replay run_user_study's per-prompt fixation points (inference.py:1010)."""
    rng = random.Random(seed)
    return [(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)) for _ in range(n)]


def load_prompts(prompt_set):
    path = os.path.join(EVAL_DIR, PROMPT_SETS[prompt_set])
    return [row["prompt"] for _, row in pd.read_csv(path).iterrows()], path


def verify_centers(args):
    """Check the replayed centers against the saved fixation_point.png of every existing
    tile. Runs on CPU, loads no model."""
    import imageio.v2 as imageio

    centers = user_study_centers(args.seed, args.num_prompts)
    worst = 0.0
    checked = 0
    for s in args.prompt_sets:
        folder = os.path.join(args.reference_root, f"baseline_comparison_randomized_{s}")
        for idx in range(args.num_prompts):
            p = os.path.join(folder, f"{idx:05d}", "fixation_point.png")
            if not os.path.exists(p):
                print(f"  missing {p}")
                continue
            im = imageio.imread(p)
            ys, xs = np.nonzero(im[..., 0] > 128)
            h, w = im.shape[:2]
            err = max(abs(xs.mean() / w - 0.5 - centers[idx][0]),
                      abs(ys.mean() / h - 0.5 - centers[idx][1]))
            worst = max(worst, err)
            checked += 1
    print(f"verify_centers: {checked} tiles, worst |center error| = {worst:.2e} "
          f"(normalized; dot rasterization alone is ~1e-4)")
    if worst > 1e-3:
        raise SystemExit("center replay does not match the saved fixation points")
    print("verify_centers: OK")


def build_masks(args, device="cpu"):
    """Per-prompt masks for both arms, from the same centers.

    Returns (centers, tome_masks, ralu_masks, full_res_masks). `tome_masks` are the (H/16,
    W/16) token grids `create_foveation_mask` hands our pipeline; `ralu_masks` are the
    (H/32, W/32) coarse block grids RALU's stage 1 works on. Same geometry either way.
    """
    from inference import create_foveation_mask, create_foveation_mask_full_res
    from RALU_fov_inference_diffsynth import coarse_foveation_mask

    centers = user_study_centers(args.seed, args.num_prompts)
    tome, ralu, full = [], [], []
    for c in centers:
        tome.append(create_foveation_mask(
            args.height, args.width, center=c, r=USER_STUDY_RADIUS, shape=USER_STUDY_SHAPE,
            device=device, lr_factor=args.lr_downsample_factor))
        ralu.append(coarse_foveation_mask(
            args.height, args.width, center=c, r=USER_STUDY_RADIUS, shape=USER_STUDY_SHAPE,
            lr_factor=args.lr_downsample_factor))
        full.append(create_foveation_mask_full_res(
            args.height, args.width, center=c, r=USER_STUDY_RADIUS, shape=USER_STUDY_SHAPE,
            device=device))
    return centers, tome, ralu, full


def ralu_schedules(args, ralu_masks):
    """Compute-matched RALU schedule for every prompt. CPU only."""
    from pipeline_flux2_diffsynth_RALU_fov import ralu_schedule, stage_token_counts

    out = []
    for mask in ralu_masks:
        m = int(mask.sum())
        N, e, info = ralu_schedule(
            total_steps=args.steps, level=args.ralu_level, num_stages=args.num_stages,
            match_compute=True, height=args.height, width=args.width,
            f0=args.lr_downsample_factor, n_hr_blocks=m, ours_steps=args.ours_steps,
        )
        out.append({
            "m_blocks": m,
            "N": N,
            "e": e,
            "tokens_per_stage": stage_token_counts(
                args.height, args.width, args.lr_downsample_factor, m, args.num_stages),
            "cost_ratio": info.get("cost_ratio"),
        })
    return out


def plan(args):
    """Print what every prompt index will run, load nothing."""
    _, tome_masks, ralu_masks, _ = build_masks(args)
    m_blocks = [int(mm.sum()) for mm in ralu_masks]
    kept = [int(mm.sum()) for mm in tome_masks]
    M = (args.height // 16) * (args.width // 16)
    M_down = M // (args.lr_downsample_factor ** 2)
    mixed = [M_down + (args.lr_downsample_factor ** 2 - 1) * m for m in m_blocks]
    print(f"masks: circular r={USER_STUDY_RADIUS} lr_factor={args.lr_downsample_factor} "
          f"at {args.height}x{args.width}")
    print(f"  fovea blocks m : min {min(m_blocks)}  max {max(m_blocks)}  "
          f"mean {np.mean(m_blocks):.1f}  (of {M_down})")
    print(f"  HR tokens      : min {min(kept)}  max {max(kept)}  (of {M})")
    print(f"  mixed tokens   : min {min(mixed)}  max {max(mixed)}  "
          f"mean {np.mean(mixed):.1f}  ({np.mean(mixed) / M:.1%} of full res)")
    print(f"  ours cost      : {args.ours_steps} x mixed = "
          f"{args.ours_steps * int(np.mean(mixed)):,} token-steps (mean)")
    if args.arm in ("ralu", "both"):
        sched = ralu_schedules(args, ralu_masks)
        totals = [sum(s["N"]) for s in sched]
        ratios = [s["cost_ratio"] for s in sched]
        uniq = sorted({tuple(s["N"]) for s in sched})
        print(f"RALU {args.num_stages}-stage level {args.ralu_level}, z_max={args.z_max}, "
              f"compute-matched to ours@{args.ours_steps}:")
        print(f"  distinct schedules : {[list(u) for u in uniq]}")
        print(f"  total steps        : min {min(totals)} max {max(totals)} "
              f"mean {np.mean(totals):.1f}")
        print(f"  cost ratio vs ours : min {min(ratios):.4f} max {max(ratios):.4f} "
              f"mean {np.mean(ratios):.4f}")
    if args.arm in ("tome", "both"):
        print(f"ToMe-2x2 merge_mlp={args.merge_mlp}: {args.steps} steps at the mixed token "
              f"count above, i.e. our budget exactly.")


def _model_configs(args, ModelConfig):
    return (
        [
            ModelConfig(model_id=args.model_id, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        ModelConfig(model_id=args.model_id, origin_file_pattern="tokenizer/"),
    )


def save_image(image, out_dir, arm, full_res_mask, args):
    """Write the plain image plus, when --outline, an outlined copy matching img_naive /
    img_ours in the existing tiles."""
    import imageio.v2 as imageio
    from inference import draw_foveation_outline

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"img_{arm}_raw.png")
    image.save(raw_path)
    if not args.outline:
        return
    img = imageio.imread(raw_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    draw_foveation_outline(
        img, full_res_mask, args.height, args.width,
        outline_width_frac=args.outline_width_frac,
        color=tuple(int(x) for x in args.outline_color.split(",")),
    )
    imageio.imwrite(os.path.join(out_dir, f"img_{arm}.png"), img)


def run_tome(args, tasks, tome_masks, full_masks):
    import torch
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
    from flux2_tome import apply_patch

    torch.manual_seed(args.seed)
    model_configs, tokenizer_config = _model_configs(args, ModelConfig)
    print("Loading stock FLUX.2 pipeline (no foveated pipeline, no TokenAE)...")
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    if args.lora_checkpoint:
        pipe.load_lora(pipe.dit, args.lora_checkpoint)
        print(f"Loaded LoRA checkpoint from {args.lora_checkpoint}")

    lh, lw = args.height // 16, args.width // 16
    times = []
    for prompt_set, idx, prompt, out_dir in tasks:
        t0 = time.time()
        # Rebinding is cheap next to a 50-step sample, so just re-patch per prompt rather
        # than mutating the shared state dict in place.
        state = apply_patch(
            pipe.dit, tome_masks[idx].to(pipe.device), lh, lw,
            lr_factor=args.lr_downsample_factor,
            merge_mlp=args.merge_mlp, enabled=True,
        )
        image = pipe(
            prompt=prompt,
            height=args.height, width=args.width,
            seed=args.seed, rand_device="cuda",
            num_inference_steps=args.steps,
            cfg_scale=args.cfg,
        )
        save_image(image, out_dir, "tome", full_masks[idx], args)
        dt = time.time() - t0
        times.append(dt)
        print(f"[set {prompt_set} {idx:05d}] tome {dt:.1f}s  "
              f"{state['merger'].num_merged}/{state['merger'].num_tokens} tokens  "
              f"{prompt[:60]}", flush=True)
    return times


def run_ralu(args, tasks, ralu_masks, full_masks):
    import torch
    from diffsynth.pipelines.flux2_image import ModelConfig
    from pipeline_flux2_diffsynth_RALU_fov import (
        Flux2DiffSynthRALUFovPipeline, ralu_schedule, stage_token_counts,
    )

    model_configs, tokenizer_config = _model_configs(args, ModelConfig)
    print("Loading FLUX.2 RALU-foveated pipeline...")
    pipe = Flux2DiffSynthRALUFovPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    if args.lora_checkpoint:
        pipe.load_lora(pipe.dit, args.lora_checkpoint)
        print(f"Loaded LoRA checkpoint from {args.lora_checkpoint}")
    pipe.id_frame = args.id_frame

    seq_len = (args.height // 16) * (args.width // 16)
    times = []
    for prompt_set, idx, prompt, out_dir in tasks:
        mask = ralu_masks[idx]
        m = int(mask.sum())
        # The mask moves with the fixation point, so m moves too: re-solve the matched
        # schedule, the NT-DM params and the correlated noise for every prompt.
        N, e, info = ralu_schedule(
            total_steps=args.steps, level=args.ralu_level, num_stages=args.num_stages,
            match_compute=True, height=args.height, width=args.width,
            f0=args.lr_downsample_factor, n_hr_blocks=m, ours_steps=args.ours_steps,
        )
        pipe.set_foveation_mask(mask)
        pipe.set_params(
            N=N, e=e, seq_len=seq_len, f0=args.lr_downsample_factor,
            z_max=args.z_max, shift_mode=args.shift_mode,
            stage_tokens=stage_token_counts(
                args.height, args.width, args.lr_downsample_factor, m, args.num_stages),
        )
        pipe.generate_ralu_noise(args.height, args.width, seed=args.seed)
        t0 = time.time()
        image = pipe(
            prompt=prompt,
            cfg_scale=args.cfg,
            embedded_guidance=args.embedded_guidance,
            height=args.height, width=args.width,
            seed=args.seed,
        )
        save_image(image, out_dir, "ralu", full_masks[idx], args)
        dt = time.time() - t0
        times.append(dt)
        print(f"[set {prompt_set} {idx:05d}] ralu {dt:.1f}s  N={N} m={m} "
              f"ratio={info.get('cost_ratio')}  {prompt[:55]}", flush=True)
    return times


def main():
    args = build_args()

    if args.verify_centers:
        verify_centers(args)
        return
    if args.plan:
        plan(args)
        return

    _, tome_masks, ralu_masks, full_masks = build_masks(args)

    # One flat task list across the requested prompt sets, sharded round-robin so every
    # worker gets a mix of all three.
    tasks, flat = [], 0
    for prompt_set in args.prompt_sets:
        prompts, csv_path = load_prompts(prompt_set)
        assert len(prompts) >= args.num_prompts, f"{csv_path} has only {len(prompts)} rows"
        out_root = os.path.join(args.out_root,
                                f"baseline_comparison_randomized_{prompt_set}_extra")
        for idx in range(args.num_prompts):
            if flat % args.num_subsets == args.subset_idx:
                tasks.append((prompt_set, idx, prompts[idx],
                              os.path.join(out_root, f"{idx:05d}")))
            flat += 1
    print(f"shard {args.subset_idx}/{args.num_subsets}: {len(tasks)} images, arm={args.arm}")

    runner = {"tome": run_tome, "ralu": run_ralu}[args.arm]
    masks = tome_masks if args.arm == "tome" else ralu_masks
    times = runner(args, tasks, masks, full_masks)

    os.makedirs(args.out_root, exist_ok=True)
    stats = {
        "arm": args.arm,
        "prompt_sets": {s: PROMPT_SETS[s] for s in args.prompt_sets},
        "num_images": len(tasks),
        "mask": {"shape": USER_STUDY_SHAPE, "r": USER_STUDY_RADIUS,
                 "lr_downsample_factor": args.lr_downsample_factor,
                 "centers": "run_user_study replay, seed %d" % args.seed},
        "steps": args.steps, "cfg": args.cfg, "seed": args.seed,
        "lora": args.lora_checkpoint,
        "merge_mlp": args.merge_mlp if args.arm == "tome" else None,
        "ralu": None if args.arm == "tome" else {
            "level": args.ralu_level, "num_stages": args.num_stages,
            "match_compute": True, "ours_steps": args.ours_steps, "z_max": args.z_max,
        },
        "mean_time_s": float(np.mean(times)) if times else None,
    }
    with open(os.path.join(args.out_root,
                           f"stats_{args.arm}_{args.subset_idx:02d}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print("Done.")


def build_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["tome", "ralu", "both"], default="tome")
    p.add_argument("--prompt_sets", type=int, nargs="+", default=[1, 2, 3],
                   choices=sorted(PROMPT_SETS))
    p.add_argument("--num_prompts", type=int, default=100)
    p.add_argument("--out_root", type=str,
                   default=os.path.join(EVAL_DIR, "../results/eccv2026_final_v2"))
    p.add_argument("--reference_root", type=str,
                   default=os.path.join(EVAL_DIR, "../results/eccv2026_final_v2"),
                   help="where the existing baseline_comparison_randomized_* live "
                        "(--verify_centers only)")
    # Model. Defaults to the BASE model: run_user_study clears LoRA for the high_res and
    # naive columns these rows sit beside.
    p.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-base-4B")
    p.add_argument("--lora_checkpoint", type=str, default=None)
    # Generation — mirrors user_study.sh, which took inference.py's defaults.
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr_downsample_factor", type=int, default=2)
    # ToMe
    p.add_argument("--merge_mlp", type=_str_to_bool, default=True)
    # RALU
    p.add_argument("--ralu_level", type=int, default=4)
    p.add_argument("--num_stages", type=int, default=3, choices=[2, 3])
    p.add_argument("--ours_steps", type=int, default=50)
    p.add_argument("--z_max", type=float, default=6.5)
    p.add_argument("--shift_mode", type=str, default="ntdm", choices=["ntdm", "natural"])
    p.add_argument("--id_frame", type=str, default="coarse", choices=["coarse", "full"])
    p.add_argument("--embedded_guidance", type=float, default=4.0)
    # Outline, matching img_naive / img_ours
    p.add_argument("--outline", type=_str_to_bool, default=True)
    p.add_argument("--outline_width_frac", type=float, default=0.005)
    p.add_argument("--outline_color", type=str, default="255,0,0")
    # Sharding
    p.add_argument("--num_subsets", type=int, default=1)
    p.add_argument("--subset_idx", type=int, default=0)
    # Dry runs
    p.add_argument("--plan", action="store_true", help="print schedules, load no model")
    p.add_argument("--verify_centers", action="store_true",
                   help="check the mask replay against the saved fixation points, then exit")
    return p.parse_args()


if __name__ == "__main__":
    main()
