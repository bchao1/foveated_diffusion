"""
Isolated latency benchmark: ToMe-2x2 vs ours vs full-resolution, at a matched token budget.

Runs ONE arm per process so only one pipeline is resident, and reports two numbers with an
identical definition across arms:

  dit_s    summed CUDA-synced wall time of every DiT forward call. This is what token
           reduction actually acts on, and it includes ToMe's per-block merge/unmerge.
  total_s  the whole pipe(...) call: text encode + denoise + VAE decode.

`total_s` is NOT apples-to-apples across arms — ours uses decode_mode="merge", which decodes
the VAE twice. Compare `dit_s`; read `total_s` as an end-to-end sanity check.

    python bench_tome_vs_ours.py --arm ours --out bench/ours.json
    python bench_tome_vs_ours.py --report bench/          # aggregate to a table

Nothing here imports or mutates DiffSynth; it reuses eval/inference.py read-only.
"""

import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ARMS = ("high_res", "tome_attn", "tome_mlp", "ours", "naive")

PROMPT = ("A lion lounges calmly, displaying its powerful physique, with a soft focus "
          "highlighting its majestic mane and intense gaze.")

NOFOV_LORA = "/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/step-15000.safetensors"
FOV_LORA = "/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-5000.safetensors"


class DiTTimer:
    """Accumulates CUDA-synced wall time across DiT forward calls, by wrapping the bound
    method on the instance. Restored on exit; the module itself is not modified."""

    def __init__(self, dit):
        self.dit = dit
        self.original = dit.forward
        self.total = 0.0
        self.calls = 0

    def __enter__(self):
        def timed(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = self.original(*a, **kw)
            torch.cuda.synchronize()
            self.total += time.perf_counter() - t0
            self.calls += 1
            return out

        self.dit.forward = timed
        return self

    def __exit__(self, *exc):
        self.dit.forward = self.original

    def reset(self):
        self.total = 0.0
        self.calls = 0


def _bench_args(a):
    """Minimal namespace for the helpers we reuse from eval/inference.py."""
    return argparse.Namespace(model_id=a.model_id)


def build(arm, a):
    """Returns (call_fn, pipe, num_tokens). call_fn() runs one full generation."""
    from inference import create_foveation_mask, create_foveation_mask_full_res

    if arm in ("high_res", "tome_attn", "tome_mlp"):
        from tome_fov_inference import load_pipeline
        pipe = load_pipeline(argparse.Namespace(
            model_id=a.model_id, model_path=a.model_path, lora_checkpoint=NOFOV_LORA))

        mask = create_foveation_mask(
            a.height, a.width, center=(0, 0), r=a.radius, shape=a.mask,
            device=pipe.device, lr_factor=a.lr_factor)

        from flux2_tome import apply_patch
        state = apply_patch(
            pipe.dit, mask, a.height // 16, a.width // 16, lr_factor=a.lr_factor,
            merge_mlp=(arm == "tome_mlp"), enabled=(arm != "high_res"))
        n_tokens = (a.height // 16) * (a.width // 16) if arm == "high_res" else state["merger"].num_merged

        def call():
            return pipe(prompt=PROMPT, height=a.height, width=a.width, seed=a.seed,
                        rand_device="cuda", num_inference_steps=a.steps, cfg_scale=a.cfg)

        return call, pipe, n_tokens

    # ours / naive: the foveated pipeline, exactly as eval/inference.py drives it
    from inference import load_pipeline as load_fov_pipeline
    pipe = load_fov_pipeline(_bench_args(a), use_foveated_pipeline=True)
    lora = FOV_LORA if arm == "ours" else NOFOV_LORA
    pipe.load_lora(pipe.dit, lora)
    print(f"Loaded LoRA checkpoint from {lora}")

    mask = create_foveation_mask(
        a.height, a.width, center=(0, 0), r=a.radius, shape=a.mask,
        device=pipe.device, lr_factor=a.lr_factor)
    full_res_mask = create_foveation_mask_full_res(
        a.height, a.width, center=(0, 0), r=a.radius, shape=a.mask, device=pipe.device)

    from flux2_tome.merge_2x2 import BlockMerger  # only to report the token count
    n_tokens = BlockMerger(mask, a.height // 16, a.width // 16, lr_factor=a.lr_factor).num_merged

    def call():
        return pipe(prompt=PROMPT, height=a.height, width=a.width, seed=a.seed,
                    rand_device="cuda", num_inference_steps=a.steps, cfg_scale=a.cfg,
                    foveation_mask=mask, full_res_foveation_mask=full_res_mask,
                    token_upsampler=None, token_downsampler=None,
                    decode_mode="merge", prediction_type="clean",
                    tokenAE_training=False, soft_foveation_blend=True,
                    lr_downsample_factor=a.lr_factor)

    return call, pipe, n_tokens


def run(a):
    call, pipe, n_tokens = build(a.arm, a)

    with DiTTimer(pipe.dit) as timer:
        print(f"[{a.arm}] warmup ({a.warmup})...")
        for _ in range(a.warmup):
            call()

        dit_s, total_s, calls = [], [], []
        for i in range(a.repeats):
            timer.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            call()
            torch.cuda.synchronize()
            total_s.append(time.perf_counter() - t0)
            dit_s.append(timer.total)
            calls.append(timer.calls)
            print(f"[{a.arm}] rep {i + 1}/{a.repeats}: dit {dit_s[-1]:.2f}s  total {total_s[-1]:.2f}s")

    result = {
        "arm": a.arm, "tokens": n_tokens, "steps": a.steps,
        "mask": a.mask, "radius": a.radius, "lr_factor": a.lr_factor,
        "dit_s_mean": sum(dit_s) / len(dit_s), "dit_s_min": min(dit_s),
        "total_s_mean": sum(total_s) / len(total_s), "total_s_min": min(total_s),
        "dit_calls": calls[0], "repeats": a.repeats,
        "gpu": torch.cuda.get_device_name(0),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def report(folder):
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        print(f"no .json results in {folder}")
        return
    order = {a: i for i, a in enumerate(ARMS)}
    rows.sort(key=lambda r: order.get(r["arm"], 99))

    base = next((r for r in rows if r["arm"] == "high_res"), None)
    print(f"\n{rows[0]['steps']} steps, {rows[0]['mask']} r={rows[0]['radius']}, "
          f"lr_factor={rows[0]['lr_factor']}, {rows[0]['repeats']} reps, {rows[0]['gpu']}\n")
    print(f"{'arm':<12} {'tokens':>7} {'DiT s':>8} {'speedup':>8} {'total s':>9} {'DiT calls':>10}")
    print("-" * 60)
    for r in rows:
        sp = f"{base['dit_s_mean'] / r['dit_s_mean']:.2f}x" if base else "-"
        print(f"{r['arm']:<12} {r['tokens']:>7} {r['dit_s_mean']:>8.2f} {sp:>8} "
              f"{r['total_s_mean']:>9.2f} {r['dit_calls']:>10}")
    print("\nDiT s = summed DiT forward time (what token reduction acts on).")
    print("total s not comparable across arms: ours decodes the VAE twice (decode_mode=merge).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--report", type=str, default=None, help="aggregate a folder of result JSONs")
    p.add_argument("--out", type=str, default="bench/result.json")
    p.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-base-4B")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--mask", type=str, default="circular")
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--lr_factor", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    if args.report:
        report(args.report)
    elif args.arm:
        run(args)
    else:
        p.error("pass --arm or --report")
