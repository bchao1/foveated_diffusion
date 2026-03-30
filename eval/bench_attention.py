"""
Standalone attention benchmark for foveated diffusion.

Tests each available attention implementation across foveation mask configurations,
measuring wall-clock time at the raw kernel level (no pipeline overhead).

Foveation model (FLUX2 latent space, 16px stride):
  - Image size H×W → latent grid (H//16) × (W//16)
  - HR region:  num_hr tokens at full resolution
  - LR region:  remaining tokens downsampled by lr_factor^2
  - Total tokens fed to attention: num_hr + (total - num_hr) // lr_factor^2

FLUX2 defaults (klein-base-4B):
  - 48 heads, head_dim=128, inner_dim=6144
  - Text tokens: 512 (T5)
  - Double-stream layers: image and text attend separately
  - Single-stream layers: [text || image] joint sequence

Usage:
  python bench_attention.py
  python bench_attention.py --plot attn_bench.png
  python bench_attention.py --height 1024 --width 1024 --dtype bfloat16
  python bench_attention.py --impls flash_attention_2 torch --token-ratios 0.25 0.5 1.0
  python bench_attention.py --warmup 3 --repeats 10 --batch-size 2
"""

import argparse
import os
import sys
import time
import math

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

# ── Import all attention variants ────────────────────────────────────────────

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ModuleNotFoundError:
    XFORMERS_AVAILABLE = False

try:
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
    SPARGE_ATTN_2_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    SPARGE_ATTN_2_AVAILABLE = False

try:
    from spas_sage_attn import spas_sage_attn_meansim_topk_cuda
    SPARGE_ATTN_1_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    SPARGE_ATTN_1_AVAILABLE = False

SPARGE_ATTN_AVAILABLE = SPARGE_ATTN_2_AVAILABLE or SPARGE_ATTN_1_AVAILABLE

AVAILABILITY = {
    "flash_attention_3": FLASH_ATTN_3_AVAILABLE,
    "flash_attention_2": FLASH_ATTN_2_AVAILABLE,
    "sage_attention":    SAGE_ATTN_AVAILABLE,
    "xformers":          XFORMERS_AVAILABLE,
    "sparge_attention":  SPARGE_ATTN_AVAILABLE,
    "torch":             True,
}

ALL_IMPLS = list(AVAILABILITY.keys())


# ── Thin wrappers matching attention.py conventions ──────────────────────────
# All wrappers accept (q, k, v) in (b, n, s, d) format (batch, heads, seq, dim)
# and return output in the same format.

def run_torch(q, k, v, scale=None):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)


def run_flash3(q, k, v, scale=None):
    # flash3 expects (b, s, n, d)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=scale)
    if isinstance(out, tuple):
        out = out[0]
    return out.transpose(1, 2)


def run_flash2(q, k, v, scale=None):
    # flash2 expects (b, s, n, d)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = flash_attn.flash_attn_func(q, k, v, softmax_scale=scale)
    return out.transpose(1, 2)


def run_sage(q, k, v, scale=None):
    return sageattn(q, k, v, sm_scale=scale)


def run_xformers(q, k, v, scale=None):
    # xformers expects (b, s, n, d)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = xops.memory_efficient_attention(q, k, v, scale=scale)
    return out.transpose(1, 2)


def run_sparge(q, k, v, scale=None, topk=0.5):
    if SPARGE_ATTN_1_AVAILABLE and _is_ampere():
        out = spas_sage_attn_meansim_topk_cuda(q, k, v, topk=topk, is_causal=False)
    else:
        out = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=topk, is_causal=False)
    return out


def _is_ampere():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return major == 8
    return False


RUNNERS = {
    "torch":             run_torch,
    "flash_attention_3": run_flash3,
    "flash_attention_2": run_flash2,
    "sage_attention":    run_sage,
    "xformers":          run_xformers,
    "sparge_attention":  run_sparge,
}

# dtype requirements per implementation
DTYPE_REQUIREMENTS = {
    "torch":             None,          # accepts fp16, bf16, fp32
    "flash_attention_3": torch.bfloat16,
    "flash_attention_2": torch.float16,
    "sage_attention":    None,          # accepts fp16, bf16
    "xformers":          None,
    "sparge_attention":  None,
}


# ── Token count model ─────────────────────────────────────────────────────────

def compute_token_counts(height, width, hr_frac, lr_factor, text_tokens):
    """
    Compute token sequence lengths for double- and single-stream layers.

    hr_frac: fraction of image area that is high-resolution (0–1).
    Returns dict with keys 'img_only' and 'joint' (img + text).
    """
    latent_h = height // 16
    latent_w = width  // 16
    total_img = latent_h * latent_w
    num_hr = int(round(total_img * hr_frac))
    num_lr = (total_img - num_hr) // (lr_factor ** 2)
    img_tokens = num_hr + num_lr
    return {
        "total_img":   total_img,
        "num_hr":      num_hr,
        "num_lr":      num_lr,
        "img_tokens":  img_tokens,
        "joint_tokens": img_tokens + text_tokens,
        "token_ratio": img_tokens / total_img,
    }


def fov_mask_from_hr_frac(height, width, hr_frac):
    """
    Return a binary foveation mask (H//16, W//16) with the top-left
    hr_frac fraction of tokens set to 1 (square patch approximation).
    """
    lh = height // 16
    lw = width  // 16
    total = lh * lw
    num_hr = int(round(total * hr_frac))
    side = int(math.ceil(math.sqrt(num_hr)))
    side = min(side, min(lh, lw))
    mask = torch.zeros(lh, lw, dtype=torch.float32)
    mask[:side, :side] = 1.0
    return mask


# ── Benchmarking ─────────────────────────────────────────────────────────────

def make_qkv(batch, heads, seq, head_dim, dtype, device):
    shape = (batch, heads, seq, head_dim)
    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device)
    return q, k, v


def benchmark_impl(impl_name, q, k, v, scale, warmup, repeats):
    runner = RUNNERS[impl_name]
    # dtype cast if required
    req_dtype = DTYPE_REQUIREMENTS[impl_name]
    if req_dtype is not None and q.dtype != req_dtype:
        q, k, v = q.to(req_dtype), k.to(req_dtype), v.to(req_dtype)

    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = runner(q, k, v, scale=scale)
    torch.cuda.synchronize()

    # timed runs
    times = []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = runner(q, k, v, scale=scale)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return times


# ── Foveation mask shapes ─────────────────────────────────────────────────────

MASK_SHAPES = {
    "square": "Top-left square patch (approximates square foveal region)",
    "circular": "Centered circular disk (realistic foveal shape)",
    "checkerboard": "Alternating HR/LR tokens (50% HR regardless of hr_frac)",
}


def hr_fracs_for_shape(shape, height, width, hr_frac):
    """
    Return (mask_label, effective_hr_frac) for a given mask shape and target hr_frac.
    For checkerboard, hr_frac is always 0.5.
    """
    if shape == "checkerboard":
        return "checkerboard", 0.5
    return shape, hr_frac


# ── Plotting ─────────────────────────────────────────────────────────────────

IMPL_STYLES = {
    "torch":             dict(color="#555555", ls="-",  marker="o", label="PyTorch SDPA"),
    "flash_attention_2": dict(color="#1f77b4", ls="-",  marker="s", label="FlashAttn 2"),
    "flash_attention_3": dict(color="#17becf", ls="-",  marker="^", label="FlashAttn 3"),
    "sage_attention":    dict(color="#2ca02c", ls="-",  marker="D", label="SageAttn"),
    "sparge_attention":  dict(color="#ff7f0e", ls="-",  marker="P", label="SpargeAttn"),
    "xformers":          dict(color="#9467bd", ls="-",  marker="X", label="xFormers"),
}


def plot_results(results, impls_to_test, args, save_path):
    """
    Three-panel figure per (mask_shape, layer_type) combination:
      Left:   Latency (ms) vs token ratio — one line per attention impl.
      Centre: Speedup vs torch vs token ratio.
      Right:  Latency (ms) vs sequence length (shows quadratic scaling).

    Each panel gets error-band shading (±1σ).
    """
    combos = [(ms, lt) for ms in args.mask_shapes for lt in args.layer_types]
    n_combos = len(combos)
    n_cols = 3
    fig_h = 4.2 * n_combos
    fig, axes = plt.subplots(n_combos, n_cols, figsize=(n_cols * 5.5, fig_h), squeeze=False)
    fig.suptitle(
        f"Attention Kernel Benchmark  —  {args.height}×{args.width}  "
        f"heads={args.num_heads}  head_dim={args.head_dim}  dtype={args.dtype}\n"
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}  "
        f"lr_factor={args.lr_factor}  batch={args.batch_size}  repeats={args.repeats}",
        fontsize=11, y=1.01,
    )

    for row_idx, (mask_shape, layer_type) in enumerate(combos):
        ax_lat, ax_spd, ax_seq = axes[row_idx]
        row_title = f"mask={mask_shape}  layer={layer_type}"

        # Gather data for this combo
        token_ratios = sorted(args.token_ratios)
        torch_means = {}

        for impl in impls_to_test:
            means, stds, seqs, tr = [], [], [], []
            for hf in token_ratios:
                r = results.get((impl, mask_shape, layer_type, hf), {})
                means.append(r.get("mean_ms", float("nan")))
                stds.append(r.get("std_ms",  float("nan")))
                seqs.append(r.get("seq",     0))
                tr.append(r.get("token_ratio", hf))
            means = np.array(means)
            stds  = np.array(stds)
            seqs  = np.array(seqs, dtype=float)
            tr    = np.array(tr)

            if impl == "torch":
                torch_means = {hf: m for hf, m in zip(token_ratios, means)}

            style = IMPL_STYLES.get(impl, dict(color="gray", ls="--", marker=".", label=impl))
            kw = dict(color=style["color"], marker=style["marker"],
                      linestyle=style["ls"], linewidth=1.8, markersize=6,
                      label=style["label"])

            # Left: latency vs token ratio
            ax_lat.plot(tr, means, **kw)
            ax_lat.fill_between(tr, means - stds, means + stds,
                                alpha=0.15, color=style["color"])

            # Right: latency vs seq length
            order = np.argsort(seqs)
            ax_seq.plot(seqs[order], means[order], **kw)
            ax_seq.fill_between(seqs[order],
                                (means - stds)[order], (means + stds)[order],
                                alpha=0.15, color=style["color"])

        # Centre: speedup vs torch
        for impl in impls_to_test:
            if impl == "torch":
                continue
            speedups = []
            for hf in token_ratios:
                r   = results.get((impl, mask_shape, layer_type, hf), {})
                tms = torch_means.get(hf, float("nan"))
                ims = r.get("mean_ms", float("nan"))
                speedups.append(tms / ims if ims > 0 else float("nan"))
            style = IMPL_STYLES.get(impl, dict(color="gray", ls="--", marker=".", label=impl))
            ax_spd.plot(
                token_ratios, speedups,
                color=style["color"], marker=style["marker"],
                linestyle=style["ls"], linewidth=1.8, markersize=6,
                label=style["label"],
            )

        # Formatting — left panel
        ax_lat.set_title(f"{row_title}\nLatency vs token ratio", fontsize=9)
        ax_lat.set_xlabel("HR fraction of image tokens")
        ax_lat.set_ylabel("Latency (ms)")
        ax_lat.legend(fontsize=8)
        ax_lat.grid(True, alpha=0.3)
        ax_lat.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        # Add secondary x-axis showing actual sequence length
        total_img = (args.height // 16) * (args.width // 16)
        def tr_to_seq(x):
            # approximate: seq ~ total_img*(x + (1-x)/lr_factor^2) [img_only]
            img_seq = total_img * (x + (1 - x) / args.lr_factor**2)
            return img_seq + (args.text_tokens if layer_type == "joint" else 0)
        ax2 = ax_lat.secondary_xaxis("top", functions=(tr_to_seq, lambda s: s))
        ax2.set_xlabel("Approx. sequence length", fontsize=8)
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

        # Formatting — centre panel
        ax_spd.axhline(1.0, color="#555555", ls="--", lw=1, alpha=0.6, label="PyTorch (1×)")
        ax_spd.set_title(f"{row_title}\nSpeedup vs PyTorch SDPA", fontsize=9)
        ax_spd.set_xlabel("HR fraction of image tokens")
        ax_spd.set_ylabel("Speedup (×)")
        ax_spd.legend(fontsize=8)
        ax_spd.grid(True, alpha=0.3)
        ax_spd.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        # Formatting — right panel
        ax_seq.set_title(f"{row_title}\nLatency vs sequence length", fontsize=9)
        ax_seq.set_xlabel("Sequence length (tokens)")
        ax_seq.set_ylabel("Latency (ms)")
        ax_seq.legend(fontsize=8)
        ax_seq.grid(True, alpha=0.3)
        ax_seq.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--height",       type=int,   default=1024)
    p.add_argument("--width",        type=int,   default=1024)
    p.add_argument("--batch-size",   type=int,   default=1)
    p.add_argument("--num-heads",    type=int,   default=48,  help="Attention heads (FLUX2 default: 48)")
    p.add_argument("--head-dim",     type=int,   default=128, help="Head dim (FLUX2 default: 128)")
    p.add_argument("--text-tokens",  type=int,   default=512, help="T5 text token count")
    p.add_argument("--lr-factor",    type=int,   default=2,   help="LR spatial downsampling factor")
    p.add_argument("--dtype",        type=str,   default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--warmup",       type=int,   default=3)
    p.add_argument("--repeats",      type=int,   default=7)
    p.add_argument(
        "--token-ratios", type=float, nargs="+",
        default=[0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        help="HR fraction of total image tokens (0–1). 1.0 = full resolution.",
    )
    p.add_argument(
        "--mask-shapes", type=str, nargs="+",
        default=["square"],
        choices=list(MASK_SHAPES.keys()),
        help="Foveation mask shapes to sweep.",
    )
    p.add_argument(
        "--layer-types", type=str, nargs="+",
        default=["img_only", "joint"],
        choices=["img_only", "joint"],
        help="img_only = double-stream (image tokens only); joint = single-stream (img+text).",
    )
    p.add_argument(
        "--impls", type=str, nargs="+",
        default=None,
        help="Attention implementations to test. Defaults to all available.",
    )
    default_results_dir = os.path.join(os.path.dirname(__file__), "../results/bench_attn")
    p.add_argument(
        "--output", type=str,
        default=os.path.join(default_results_dir, "attn_bench.npz"),
        help="Save results as .npz to this path.",
    )
    p.add_argument(
        "--plot", type=str,
        default=os.path.join(default_results_dir, "attn_bench.png"),
        help="Save benchmark plots to this PNG path.",
    )
    p.add_argument("--no-cuda-graphs", action="store_true", help="Disable CUDA graphs (timing is raw kernel time).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = getattr(torch, args.dtype)
    scale  = 1.0 / math.sqrt(args.head_dim)

    # ── Print availability ────────────────────────────────────────────────────
    print("\n=== Attention implementation availability ===")
    for name, avail in AVAILABILITY.items():
        status = "YES" if avail else "no"
        print(f"  {name:<22} {status}")

    # ── Determine which impls to test ─────────────────────────────────────────
    if args.impls:
        impls_to_test = []
        for impl in args.impls:
            if impl not in AVAILABILITY:
                print(f"Warning: unknown impl '{impl}', skipping")
            elif not AVAILABILITY[impl]:
                print(f"Warning: '{impl}' not available, skipping")
            else:
                impls_to_test.append(impl)
    else:
        impls_to_test = [k for k, v in AVAILABILITY.items() if v]

    if not impls_to_test:
        print("No implementations available. Exiting.")
        return

    print(f"\nTesting: {impls_to_test}")
    print(f"Device:  {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"dtype:   {args.dtype}")
    print(f"Image:   {args.height}×{args.width}  heads={args.num_heads}  head_dim={args.head_dim}")
    print(f"warmup={args.warmup}  repeats={args.repeats}  batch={args.batch_size}")

    # ── Storage for results ───────────────────────────────────────────────────
    results = {}  # (impl, mask_shape, layer_type, hr_frac) -> {"mean": ..., "std": ..., "seq": ...}

    # ── Sweep ────────────────────────────────────────────────────────────────
    for mask_shape in args.mask_shapes:
        for hr_frac in args.token_ratios:
            shape_label, eff_hr_frac = hr_fracs_for_shape(mask_shape, args.height, args.width, hr_frac)
            counts = compute_token_counts(
                args.height, args.width, eff_hr_frac, args.lr_factor, args.text_tokens
            )

            for layer_type in args.layer_types:
                seq = counts["joint_tokens"] if layer_type == "joint" else counts["img_tokens"]

                header = (
                    f"\n--- mask={mask_shape}  hr_frac={hr_frac:.3f}"
                    f"  layer={layer_type}  seq={seq}"
                    f"  (token_ratio={counts['token_ratio']:.3f}"
                    f"  HR={counts['num_hr']}  LR={counts['num_lr']}) ---"
                )
                print(header)

                q, k, v = make_qkv(args.batch_size, args.num_heads, seq, args.head_dim, dtype, device)

                row = []
                for impl in impls_to_test:
                    try:
                        times = benchmark_impl(impl, q, k, v, scale, args.warmup, args.repeats)
                        mean_ms = np.mean(times) * 1000
                        std_ms  = np.std(times)  * 1000
                        print(f"  {impl:<22}  {mean_ms:7.2f} ms  ±{std_ms:5.2f} ms")
                        results[(impl, mask_shape, layer_type, hr_frac)] = {
                            "mean_ms": mean_ms, "std_ms": std_ms,
                            "seq": seq, "token_ratio": counts["token_ratio"],
                        }
                        row.append((impl, mean_ms))
                    except Exception as e:
                        print(f"  {impl:<22}  ERROR: {e}")
                        results[(impl, mask_shape, layer_type, hr_frac)] = {
                            "mean_ms": float("nan"), "std_ms": float("nan"),
                            "seq": seq, "token_ratio": counts["token_ratio"],
                        }

                # Relative speedup vs torch
                if row:
                    baseline = next((t for n, t in row if n == "torch"), None)
                    if baseline and len(row) > 1:
                        print("  Speedup vs torch:")
                        for name, t in row:
                            if name != "torch" and t > 0:
                                print(f"    {name:<22}  {baseline/t:.2f}×")

    # ── Summary table: seq-length vs impl at each mask/layer combo ───────────
    print("\n\n=== SUMMARY: mean latency (ms) ===")
    for mask_shape in args.mask_shapes:
        for layer_type in args.layer_types:
            print(f"\nmask={mask_shape}  layer={layer_type}")
            header_cols = ["hr_frac", "seq", "token_ratio"] + [f"{i:>18}" for i in impls_to_test]
            print("  " + "  ".join(header_cols))
            for hr_frac in args.token_ratios:
                row_data = results.get((impls_to_test[0], mask_shape, layer_type, hr_frac), {})
                seq   = row_data.get("seq", "?")
                tratio = row_data.get("token_ratio", float("nan"))
                cells = [f"{hr_frac:.3f}", str(seq), f"{tratio:.3f}"]
                for impl in impls_to_test:
                    r = results.get((impl, mask_shape, layer_type, hr_frac), {})
                    ms = r.get("mean_ms", float("nan"))
                    cells.append(f"{ms:>18.2f}")
                print("  " + "  ".join(cells))

    # ── Optional .npz save ───────────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        save_data = {}
        for (impl, mshape, ltype, hfrac), v in results.items():
            key = f"{impl}__{mshape}__{ltype}__{hfrac:.3f}"
            save_data[key + "__mean_ms"]     = v["mean_ms"]
            save_data[key + "__std_ms"]      = v["std_ms"]
            save_data[key + "__seq"]         = v["seq"]
            save_data[key + "__token_ratio"] = v["token_ratio"]
        np.savez(args.output, **save_data)
        print(f"\nResults saved to {args.output}")

    # ── Optional plot ────────────────────────────────────────────────────────
    if args.plot:
        os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
        plot_results(results, impls_to_test, args, args.plot)


if __name__ == "__main__":
    main()
