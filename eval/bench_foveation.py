"""
Foveated diffusion end-to-end pipeline benchmark.

Loads the FLUX2 foveated pipeline once and sweeps over foveation mask
configurations, measuring wall-clock latency for the full generation pipeline.
Uses flash_attention_2 throughout.

Mask sweep:
  - Circular and square masks at a range of radii (all centered)
  - Full-resolution baseline (no foveation, mask=None)
  - Checkerboard mask (fixed 50% HR tokens)

For each configuration records:
  - Total generation time (all denoising steps + decode)
  - Per-step time estimate
  - Token ratio (HR + LR) / total
  - Speedup vs full-resolution baseline

Results saved to ../results/bench_foveation/
  bench_foveation.npz   — raw timing arrays
  bench_foveation.png   — latency / speedup plots
  images/               — one sample image per config

Usage:
  conda run -n imp_tome python bench_foveation.py
  conda run -n imp_tome python bench_foveation.py --model-id black-forest-labs/FLUX.2-klein-base-4B
  conda run -n imp_tome python bench_foveation.py --steps 10 --repeats 3 --radii 0.2 0.4 0.6 0.8 1.0
  conda run -n imp_tome python bench_foveation.py --lora-checkpoint /path/to/lora.safetensors
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Force flash_attention_2 for the entire run
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "flash_attention_2")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../image_generation"))

from diffsynth.pipelines.flux2_image_foveated import Flux2FoveatedImagePipeline
from diffsynth.pipelines.flux2_image import ModelConfig

# Reuse mask creation logic from inference.py
from inference import create_foveation_mask

# ── Defaults ──────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results/bench_foveation")
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
DEFAULT_PROMPT = (
    "Documentary-style imagery: a lively little dog stands still on a lush green lawn, "
    "filling the entire frame. The dog has brownish-yellow fur, with both ears perked up, "
    "and an expression that is focused and cheerful. Sunlight evenly illuminates its body, "
    "making the fur appear soft, glossy, and highly detailed."
)
DEFAULT_RADII = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LR_FACTOR = 2


# ── Token ratio helper ────────────────────────────────────────────────────────

def token_stats(mask: torch.Tensor, lr_factor: int = LR_FACTOR):
    """Return (num_hr, num_lr, total_orig, token_ratio) for a token-grid mask."""
    total = mask.numel()
    hr = int(mask.sum().item())
    lr = (total - hr) // (lr_factor ** 2)
    ratio = (hr + lr) / total
    return hr, lr, total, ratio


# ── Pipeline loader ───────────────────────────────────────────────────────────

def load_pipeline(model_id: str, lora_checkpoint=None, dit_checkpoint=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading pipeline from {model_id} on {device} ...")
    pipe = Flux2FoveatedImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id=model_id, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=model_id, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id=model_id, origin_file_pattern="tokenizer/"),
    )
    if lora_checkpoint is not None:
        pipe.load_lora(pipe.dit, lora_checkpoint)
        print(f"Loaded LoRA from {lora_checkpoint}")
    if dit_checkpoint is not None:
        from diffsynth.core import load_state_dict
        state_dict = load_state_dict(dit_checkpoint, torch_dtype=torch.bfloat16)
        pipe.dit.load_state_dict(state_dict)
        print(f"Loaded DiT checkpoint from {dit_checkpoint}")
    return pipe


# ── Single timed run ──────────────────────────────────────────────────────────

def timed_run(pipe, mask, args):
    """Run pipeline once with the given foveation mask, return (image, elapsed_seconds)."""
    kwargs = dict(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        rand_device="cuda",
        num_inference_steps=args.steps,
        cfg_scale=args.guidance_scale,
        foveation_mask=mask,
        token_upsampler=None,
        token_downsampler=None,
        decode_mode="direct",
        prediction_type="clean",
        tokenAE_training=False,
        soft_foveation_blend=False,
        lr_downsample_factor=LR_FACTOR,
    )
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    image = pipe(**kwargs)
    torch.cuda.synchronize()
    return image, time.perf_counter() - t0


# ── Benchmark one configuration ───────────────────────────────────────────────

def benchmark_config(pipe, mask, label, args):
    """Warm up once, then time args.repeats runs. Returns dict of stats."""
    print(f"\n  [{label}] warming up ...")
    _, _ = timed_run(pipe, mask, args)

    times = []
    print(f"  [{label}] timing {args.repeats} repeats ...")
    image = None
    for i in range(args.repeats):
        img, t = timed_run(pipe, mask, args)
        times.append(t)
        if i == 0:
            image = img
        print(f"    rep {i+1}/{args.repeats}: {t:.2f}s")

    mean, std = float(np.mean(times)), float(np.std(times))
    print(f"  [{label}] mean={mean:.2f}s  std={std:.2f}s")
    return {"mean_s": mean, "std_s": std, "times": times, "image": image}


# ── Build sweep configs ───────────────────────────────────────────────────────

def build_configs(args):
    """
    Returns list of dicts:
      label, mask (or None), shape, radius, hr, lr, total, token_ratio
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = []

    # Full-resolution baseline (no foveation)
    configs.append({
        "label": "full_res (baseline)",
        "mask": None,
        "shape": "none",
        "radius": 1.0,
        "hr": (args.height // 16) * (args.width // 16),
        "lr": 0,
        "total": (args.height // 16) * (args.width // 16),
        "token_ratio": 1.0,
    })

    # Circular and square masks at each radius
    for shape in args.shapes:
        for r in args.radii:
            mask = create_foveation_mask(
                args.height, args.width,
                center=(0.0, 0.0), r=r, shape=shape,
                device=device, lr_factor=LR_FACTOR,
            )
            hr, lr, total, tratio = token_stats(mask, LR_FACTOR)
            label = f"{shape}  r={r:.2f}  tok={tratio:.3f}"
            configs.append({
                "label": label,
                "mask": mask,
                "shape": shape,
                "radius": r,
                "hr": hr, "lr": lr, "total": total,
                "token_ratio": tratio,
            })

    # Checkerboard (fixed 50% HR)
    if args.checkerboard:
        mask = create_foveation_mask(
            args.height, args.width,
            center=(0.0, 0.0), r=0.5, shape="checkerboard",
            device=device, lr_factor=LR_FACTOR,
        )
        hr, lr, total, tratio = token_stats(mask, LR_FACTOR)
        configs.append({
            "label": f"checkerboard  tok={tratio:.3f}",
            "mask": mask,
            "shape": "checkerboard",
            "radius": 0.5,
            "hr": hr, "lr": lr, "total": total,
            "token_ratio": tratio,
        })

    return configs


# ── Plotting ──────────────────────────────────────────────────────────────────

SHAPE_STYLE = {
    "none":         dict(color="#555555", marker="*", ls="--", label="Full-res baseline", zorder=5),
    "circular":     dict(color="#1f77b4", marker="o", ls="-",  label="Circular mask"),
    "square":       dict(color="#ff7f0e", marker="s", ls="-",  label="Square mask"),
    "checkerboard": dict(color="#2ca02c", marker="D", ls=":",  label="Checkerboard"),
}


def plot_results(configs, timing, args, save_path):
    """
    Four-panel figure:
      Top-left:    Latency (s) vs token ratio — per shape
      Top-right:   Speedup vs full-res baseline vs token ratio
      Bottom-left: Latency (s) vs mask radius — per shape
      Bottom-right: Per-step latency (ms) vs token ratio
    """
    baseline_mean = timing["full_res (baseline)"]["mean_s"]
    baseline_std  = timing["full_res (baseline)"]["std_s"]

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Foveated Diffusion Pipeline Benchmark  —  {args.height}×{args.width}  "
        f"{args.steps} steps  seed={args.seed}\n"
        f"GPU: {gpu_name}   attn: flash_attention_2   model: {args.model_id.split('/')[-1]}",
        fontsize=10, y=1.01,
    )
    ax_lat, ax_spd, ax_rad, ax_step = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Group configs by shape for clean line plots
    shape_data = {}  # shape -> sorted list of (token_ratio, radius, mean_s, std_s)
    for cfg in configs:
        shape = cfg["shape"]
        label = cfg["label"]
        t = timing[label]
        shape_data.setdefault(shape, []).append(
            (cfg["token_ratio"], cfg["radius"], t["mean_s"], t["std_s"])
        )

    for shape, points in shape_data.items():
        points.sort()  # sort by token_ratio
        trs   = np.array([p[0] for p in points])
        rads  = np.array([p[1] for p in points])
        means = np.array([p[2] for p in points])
        stds  = np.array([p[3] for p in points])
        speedups = baseline_mean / means

        style = SHAPE_STYLE.get(shape, dict(color="gray", marker=".", ls="-", label=shape))
        kw = dict(color=style["color"], marker=style["marker"],
                  linestyle=style["ls"], linewidth=2, markersize=7,
                  label=style["label"])

        # Top-left: latency vs token ratio
        ax_lat.plot(trs, means, **kw)
        ax_lat.fill_between(trs, means - stds, means + stds, alpha=0.15, color=style["color"])

        # Top-right: speedup vs token ratio
        if shape != "none":
            ax_spd.plot(trs, speedups, **kw)
            ax_spd.fill_between(
                trs,
                baseline_mean / (means + stds),
                baseline_mean / (means - stds + 1e-9),
                alpha=0.12, color=style["color"],
            )

        # Bottom-left: latency vs radius (skip baseline & checkerboard, no meaningful radius)
        if shape not in ("none", "checkerboard"):
            ax_rad.plot(rads, means, **kw)
            ax_rad.fill_between(rads, means - stds, means + stds, alpha=0.15, color=style["color"])

        # Bottom-right: per-step latency vs token ratio
        per_step_ms = means / args.steps * 1000
        per_step_std_ms = stds / args.steps * 1000
        ax_step.plot(trs, per_step_ms, **kw)
        ax_step.fill_between(trs, per_step_ms - per_step_std_ms, per_step_ms + per_step_std_ms,
                             alpha=0.15, color=style["color"])

    # Baseline annotations
    for ax in (ax_lat, ax_step):
        ref = baseline_mean if ax is ax_lat else baseline_mean / args.steps * 1000
        ax.axhline(ref, color="#555555", ls="--", lw=1.2, alpha=0.7, label="Full-res baseline")
        ax.fill_between(
            ax.get_xlim() if ax.get_xlim() != (0, 1) else [0, 1],
            ref - (baseline_std if ax is ax_lat else baseline_std / args.steps * 1000),
            ref + (baseline_std if ax is ax_lat else baseline_std / args.steps * 1000),
            alpha=0.08, color="#555555",
        )

    ax_spd.axhline(1.0, color="#555555", ls="--", lw=1.2, alpha=0.7, label="Full-res (1×)")

    # Format top-left
    ax_lat.set_title("End-to-end latency vs token ratio", fontsize=10)
    ax_lat.set_xlabel("Token ratio  (HR + LR) / total")
    ax_lat.set_ylabel("Latency (s)")
    ax_lat.legend(fontsize=8)
    ax_lat.grid(True, alpha=0.3)
    ax_lat.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # Format top-right
    ax_spd.set_title("Speedup vs full-resolution baseline", fontsize=10)
    ax_spd.set_xlabel("Token ratio  (HR + LR) / total")
    ax_spd.set_ylabel("Speedup (×)")
    ax_spd.legend(fontsize=8)
    ax_spd.grid(True, alpha=0.3)
    ax_spd.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # Format bottom-left
    ax_rad.set_title("Latency vs foveal radius", fontsize=10)
    ax_rad.set_xlabel("Mask radius (normalised)")
    ax_rad.set_ylabel("Latency (s)")
    ax_rad.legend(fontsize=8)
    ax_rad.grid(True, alpha=0.3)
    # add secondary x-axis: token ratio approximation
    total_img = (args.height // 16) * (args.width // 16)
    def r_to_tr(r):
        # For circular: token_ratio ≈ r^2 + (1 - r^2) / lr_factor^2
        return r**2 + (1 - r**2) / LR_FACTOR**2
    ax_rad2 = ax_rad.secondary_xaxis("top", functions=(r_to_tr, lambda t: t))
    ax_rad2.set_xlabel("Approx. token ratio", fontsize=8)
    ax_rad2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # Format bottom-right
    ax_step.set_title("Per-step latency vs token ratio", fontsize=10)
    ax_step.set_xlabel("Token ratio  (HR + LR) / total")
    ax_step.set_ylabel("Per-step latency (ms)")
    ax_step.legend(fontsize=8)
    ax_step.grid(True, alpha=0.3)
    ax_step.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")
    plt.close(fig)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(configs, timing, args):
    baseline_mean = timing["full_res (baseline)"]["mean_s"]
    print("\n\n=== SUMMARY ===")
    print(f"{'label':<45}  {'tok_ratio':>9}  {'mean_s':>7}  {'std_s':>6}  {'speedup':>7}  {'ms/step':>8}")
    print("-" * 95)
    for cfg in configs:
        t = timing[cfg["label"]]
        spd = baseline_mean / t["mean_s"]
        msstep = t["mean_s"] / args.steps * 1000
        print(
            f"{cfg['label']:<45}  {cfg['token_ratio']:>9.3f}  "
            f"{t['mean_s']:>7.2f}  {t['std_s']:>6.3f}  {spd:>7.2f}×  {msstep:>8.1f}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id",    default=DEFAULT_MODEL_ID)
    p.add_argument("--lora-checkpoint", default=None)
    p.add_argument("--dit-checkpoint",  default=None)
    p.add_argument("--height",  type=int,   default=1024)
    p.add_argument("--width",   type=int,   default=1024)
    p.add_argument("--steps",   type=int,   default=28,  help="Denoising steps")
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--prompt",  type=str,   default=DEFAULT_PROMPT)
    p.add_argument("--repeats", type=int,   default=3,   help="Timed repeats per config (after 1 warmup)")
    p.add_argument(
        "--radii", type=float, nargs="+", default=DEFAULT_RADII,
        help="Mask radii to sweep (0–1, normalised)",
    )
    p.add_argument(
        "--shapes", type=str, nargs="+", default=["circular", "square"],
        choices=["circular", "square"],
        help="Mask shapes to sweep",
    )
    p.add_argument("--checkerboard", action="store_true", default=True,
                   help="Also benchmark checkerboard mask")
    p.add_argument("--no-checkerboard", dest="checkerboard", action="store_false")
    p.add_argument(
        "--output-dir", default=RESULTS_DIR,
        help="Directory for results",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"\nAttention implementation: {os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION')}")
    pipe = load_pipeline(args.model_id, args.lora_checkpoint, args.dit_checkpoint)

    configs = build_configs(args)
    print(f"\n{len(configs)} configurations to benchmark:")
    for cfg in configs:
        print(f"  {cfg['label']:<50}  token_ratio={cfg['token_ratio']:.3f}  HR={cfg['hr']}  LR={cfg['lr']}")

    timing = {}
    for cfg in configs:
        result = benchmark_config(pipe, cfg["mask"], cfg["label"], args)
        timing[cfg["label"]] = result
        # Save sample image
        safe_label = cfg["label"].replace(" ", "_").replace("/", "-").replace("=", "")
        if result["image"] is not None:
            result["image"].save(os.path.join(img_dir, f"{safe_label}.png"))

    print_summary(configs, timing, args)

    # Save .npz
    npz_path = os.path.join(args.output_dir, "bench_foveation.npz")
    save_data = {}
    for cfg in configs:
        key = cfg["label"].replace(" ", "_").replace("/", "-").replace("=", "")
        t = timing[cfg["label"]]
        save_data[key + "__mean_s"]     = t["mean_s"]
        save_data[key + "__std_s"]      = t["std_s"]
        save_data[key + "__times"]      = np.array(t["times"])
        save_data[key + "__token_ratio"] = cfg["token_ratio"]
        save_data[key + "__radius"]     = cfg["radius"]
        save_data[key + "__shape"]      = cfg["shape"]
    np.savez(npz_path, **save_data)
    print(f"\nResults saved to {npz_path}")

    # Plot
    plot_path = os.path.join(args.output_dir, "bench_foveation.png")
    plot_results(configs, timing, args, plot_path)


if __name__ == "__main__":
    main()
