"""
Generate publication-quality figures for foveated diffusion paper.
Academic style: faint dashed grids, serif fonts, colorblind-friendly palette.
"""

import ast
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from argparse import ArgumentParser
import pandas as pd

# Academic figure style: serif font, faint grid, restrained colors (font sizes ×1.2)
FIGURE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
}

# Refined academic palette (colorblind-friendly, print-safe)
COLORS = {
    "primary": "#1f77b4",      # classic blue
    "secondary": "#2ca02c",   # green
    "accent": "#d62728",       # red
    "neutral_dark": "#4a4a4a",
    "fill_alpha": 0.15,
}


def setup_academic_axes(ax, grid=True):
    """Apply faint dashed grid and minor ticks in academic style."""
    if grid:
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, color="#888888", alpha=0.4)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#aaaaaa", alpha=0.25)
        ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)


def r_to_token_percentage(r):
    """Token ratio (fraction of original tokens): r^2 + (1 - r^2) / 4. Returns value in [0, 1]."""
    return r ** 2 + (1 - r ** 2) / 4


def _load_runtime_curve(npz_path):
    """Load r_list and runtime_list from npz, return (x, y) for plotting (token % vs speedup). Same logic as single-file runtime plot."""
    data = np.load(npz_path, allow_pickle=True)
    #r_list = np.asarray(data["r_list"]).flatten()
    print(data.keys())
    runtime_list = np.asarray(data["runtime_list"]).flatten()
    token_ratio_list = np.asarray(data["token_ratio_list"]).flatten()
    if runtime_list.shape[0] != token_ratio_list.shape[0]:
        raise ValueError(
            f"Mismatch in {npz_path}: runtime_list length {runtime_list.shape[0]}, token_ratio_list length {token_ratio_list.shape[0]}"
        )
    token_pct = token_ratio_list * 100
    baseline_runtime = np.asarray(runtime_list)[np.argmax(token_pct)]
    speedup = baseline_runtime / np.asarray(runtime_list)
    order = np.argsort(token_pct)
    x = token_pct[order]
    y = speedup[order]
    return x, y


def generate_runtime_figure(base_dir, output_path=None):
    """
    Plot speedup vs. token budget (% of original tokens) from all runtime_and_radius_list_{mode}.npz in base_dir.
    X = token percentage (ratio = r^2 + (1-r^2)/4), displayed decreasing left-to-right.
    Y = speedup factor vs. full-resolution (r=1) generation (baseline_runtime / runtime).
    Each file is one curve; mode (from filename) is used as the legend label.
    """
    #pattern = os.path.join(base_dir, "runtime_and_radius_list_*.npz")
    #files = sorted(glob.glob(pattern))
    files = [
        os.path.join(base_dir, "runtime_and_radius_list_image.npz"),
        os.path.join(base_dir, "runtime_and_radius_list_video.npz"),
    ]
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern}. "
            "Run the runtime experiment first (e.g. inference.py --experiment runtime) for each mode."
        )
    prefix = "runtime_and_radius_list_"
    mode_color_order = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

    with plt.rc_context(FIGURE_RC):
        fig, ax = plt.subplots(figsize=(4, 4))
        setup_academic_axes(ax, grid=False)

        x_min_global = np.inf
        x_max_global = -np.inf

        for idx, npz_path in enumerate(files):
            basename = os.path.basename(npz_path)
            stem = os.path.splitext(basename)[0]
            if stem.startswith(prefix):
                mode = stem[len(prefix):]
            else:
                mode = stem
            x, y = _load_runtime_curve(npz_path)
            color = mode_color_order[idx % len(mode_color_order)]
            ax.plot(
                x, y,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=4,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=mode,
                zorder=3,
            )
            x_min_global = min(x_min_global, x.min())
            x_max_global = max(x_max_global, x.max())

        ax.set_xlabel("Tokens (% of original)")
        ax.set_ylabel("Speedup (vs. full resolution)")
        ax.set_title("Inference speedup vs. token budget")
        ax.set_xlim(x_min_global - 1, x_max_global + 1)
        ax.set_ylim(bottom=1.0)
        ax.invert_xaxis()
        xticks = np.linspace(x_min_global, x_max_global, 4)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t:.0f}" for t in xticks])
        ax.legend(loc="upper left", framealpha=0.95, edgecolor="#cccccc")

        fig.tight_layout(pad=1.0)

        if output_path is None:
            output_path = os.path.join(base_dir, "runtime_figure.pdf")
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        print(f"Saved: {output_path}")


def generate_runtime_quality_two_col(runtime_and_radius_list_path, output_path=None):
    """
    Two-column figure: left = speedup vs token %, right = dummy image quality vs token %.
    Same format (academic style, x = tokens % decreasing left-to-right). Quality has only slight drop as x increases.
    """
    if not os.path.isfile(runtime_and_radius_list_path):
        raise FileNotFoundError(
            f"Data file not found: {runtime_and_radius_list_path}. "
            "Run the runtime experiment first (e.g. inference.py --experiment runtime)."
        )
    with plt.rc_context(FIGURE_RC):
        data = np.load(runtime_and_radius_list_path, allow_pickle=True)
        r_list = np.asarray(data["r_list"]).flatten()
        runtime_list = np.asarray(data["runtime_list"]).flatten()
        if r_list.shape[0] != runtime_list.shape[0]:
            raise ValueError(
                f"Mismatch: r_list length {r_list.shape[0]}, runtime_list length {runtime_list.shape[0]}"
            )

        token_ratio = r_to_token_percentage(r_list)
        token_pct = token_ratio * 100
        order = np.argsort(token_pct)
        x = token_pct[order]

        # Left: speedup (same as runtime figure)
        baseline_runtime = np.asarray(runtime_list)[np.argmax(token_pct)]
        speedup = baseline_runtime / np.asarray(runtime_list)
        y_speedup = speedup[order]

        # Right: dummy image quality — two lines
        # Line 1: slight drop (0.97 at 100% -> ~0.90 at 25%)
        # Line 2: more drastic drop (0.97 at 100% -> ~0.68 at 25%)
        quality_at_full = 0.97
        token_ratio_sorted = token_ratio[order]
        y_quality_slight = 0.90 + (quality_at_full - 0.90) * token_ratio_sorted
        y_quality_drastic = 0.68 + (quality_at_full - 0.68) * token_ratio_sorted

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8, 4))
        for ax in (ax_left, ax_right):
            setup_academic_axes(ax, grid=False)
            ax.set_xlabel("Tokens (% of original)")
            ax.set_xlim(x.min() - 1, x.max() + 1)
            ax.invert_xaxis()

        # Left: speedup (line only)
        ax_left.plot(
            x, y_speedup,
            color=COLORS["primary"],
            linewidth=2.0,
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="Speedup",
            zorder=3,
        )
        ax_left.set_ylabel("Speedup (vs. full resolution)")
        ax_left.set_title("Runtime")
        ax_left.set_ylim(bottom=1.0)

        # Right: two quality lines (line only)
        ax_right.plot(
            x, y_quality_slight,
            color=COLORS["secondary"],
            linewidth=2.0,
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="Ours",
            zorder=3,
        )
        ax_right.plot(
            x, y_quality_drastic,
            color=COLORS["accent"],
            linewidth=2.0,
            marker="s",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="Baseline",
            zorder=3,
        )
        ax_right.set_ylabel("Image quality")
        ax_right.set_title("Image quality")
        ax_right.set_ylim(0.65, 1.02)
        ax_right.legend(loc="lower right", framealpha=0.95, edgecolor="#cccccc")

        fig.tight_layout(pad=1.0)

        if output_path is None:
            base = os.path.splitext(runtime_and_radius_list_path)[0]
            output_path = os.path.join(os.path.dirname(base), "runtime_quality_figure.pdf")
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        print(f"Saved: {output_path}")


def generate_runtime_joint_figure(base_dir, output_path=None, fov_deg=15.0, fov_max=180.0):
    """
    Three-panel runtime figure for lr_2, lr_4, lr_8 (runtime_apollo_v2 layout).

    Panel 1 (original): Token % (inverted x) vs Speedup.
    Panel 2: Visual field FOV (degrees) vs Speedup.
    Panel 3: Visual field FOV (degrees) vs Token % of original.

    FOV is computed assuming the high-res foveal region always subtends `fov_deg`
    degrees.  Given radius r and the maximum possible radius max_r (full image),
    the full-image full FoV is:  FOV = fov_deg * max_r / r.

    Theoretical maximum speedups (1/min_token_ratio):
      lr=2 → 4×,  lr=4 → 16×,  lr=8 → 64×
    """
    LR_CONFIGS = [
        (2, COLORS["primary"],   "2× downsampling"),
        (4, COLORS["secondary"], "4× downsampling"),
        (8, COLORS["accent"],    "8× downsampling"),
    ]
    # Minimum token ratio when ALL periphery is at LR resolution: (1/lr)^2
    MIN_TOKEN = {2: 1 / 4, 4: 1 / 16, 8: 1 / 64}
    markers = {2: "o", 4: "s", 8: "^"}

    curves = {}
    for lr, color, label in LR_CONFIGS:
        npz_path = os.path.join(base_dir, f"lr_{lr}", "runtime_and_radius_list.npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Missing: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        runtime_list = np.asarray(data["runtime_list"]).flatten()
        token_ratio = np.asarray(data["token_ratio_list"]).flatten()
        r_arr = np.asarray(data["r_list"]).flatten()
        token_pct = token_ratio * 100
        baseline_runtime = runtime_list[np.argmax(token_pct)]
        speedup = baseline_runtime / runtime_list

        # ── Token % vs Speedup (original panel) ──────────────────────────────
        order = np.argsort(token_pct)

        # ── FOV-indexed curves ────────────────────────────────────────────────
        # Deduplicate: for each unique r, average runtime and token_ratio
        max_r = int(r_arr.max())
        unique_r = np.unique(r_arr)
        runtime_dedup = np.array([runtime_list[r_arr == rv].mean() for rv in unique_r])
        token_dedup   = np.array([token_ratio[r_arr == rv].mean()   for rv in unique_r])
        speedup_dedup = baseline_runtime / runtime_dedup

        # Exclude r=0 (full LR — FOV would be infinite)
        valid = unique_r > 0
        fov_arr    = fov_deg * max_r / unique_r[valid]   # degrees
        fov_speedup    = speedup_dedup[valid]
        fov_token_pct  = token_dedup[valid] * 100

        # Sort by ascending FOV, then clamp at 180°
        fov_order = np.argsort(fov_arr)
        fov_arr       = fov_arr[fov_order]
        fov_speedup   = fov_speedup[fov_order]
        fov_token_pct = fov_token_pct[fov_order]
        clamp = fov_arr <= fov_max
        fov_arr       = fov_arr[clamp]
        fov_speedup   = fov_speedup[clamp]
        fov_token_pct = fov_token_pct[clamp]

        curves[lr] = dict(
            x=token_pct[order], y=speedup[order],
            fov=fov_arr, fov_speedup=fov_speedup, fov_token_pct=fov_token_pct,
            color=color, label=label, max_r=max_r,
        )

    # Partition boundaries (%)
    b8 = MIN_TOKEN[8] * 100   # 1.5625
    b4 = MIN_TOKEN[4] * 100   # 6.25
    b2 = MIN_TOKEN[2] * 100   # 25.0

    # Derive per-panel output paths from the base output_path
    if output_path is None:
        output_path = os.path.join(base_dir, "runtime_joint_figure.pdf")
    base, ext = os.path.splitext(output_path)
    panel_paths = [
        output_path,                        # panel 1: token % vs speedup
        f"{base}_fov_speedup{ext}",         # panel 2: FOV vs speedup
        f"{base}_fov_tokens{ext}",          # panel 3: FOV vs token %
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    SAVE_KW = dict(dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    W169, H169 = 16/2.0, 9/2.0   # 16:9 figure size (inches)
    fov_all = np.concatenate([curves[lr]["fov"] for lr in (2, 4, 8)])  # already clamped

    with plt.rc_context(FIGURE_RC):

        # ══════════════════════════════════════════════════════════════════════
        # Figure 1 — Token % vs Speedup (original)
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(W169, H169))
        setup_academic_axes(ax, grid=False)

        ax.axvspan(b8, b4,  alpha=0.10, color=COLORS["accent"],    zorder=0)
        ax.axvspan(b4, b2,  alpha=0.08, color=COLORS["secondary"], zorder=0)
        ax.axvspan(b2, 101, alpha=0.06, color="#4488ff",           zorder=0)

        ax.text((b8 + b4) / 2, 0.97, "8×",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color=COLORS["accent"], style="italic")
        ax.text((b4 + b2) / 2, 0.97, "4× / 8×",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color=COLORS["secondary"], style="italic")
        ax.text((b2 + 100) / 2, 0.97, "2× / 4× / 8×",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color="#4466cc", style="italic")

        for lr, color, label in LR_CONFIGS:
            c = curves[lr]
            ax.plot(c["x"], c["y"], color=c["color"], linewidth=2.0,
                    marker=markers[lr], markersize=4,
                    markeredgecolor="white", markeredgewidth=0.7, label=label, zorder=3)

        for lr, xb, theoretical in [(8, b8, 64)]:
            color = curves[lr]["color"]
            ax.annotate(f"{theoretical}×", xy=(xb, theoretical),
                        xytext=(xb + 1.5, theoretical), fontsize=7.5, color=color,
                        va="center", arrowprops=dict(arrowstyle="-", color=color, lw=0.6), zorder=4)

        ax.set_xlabel("Tokens (% of original)")
        ax.set_ylabel("Speedup (vs. full resolution)")
        ax.set_title("Speedup vs. token count")
        x_all = np.concatenate([curves[lr]["x"] for lr in (2, 4, 8)])
        ax.set_xlim(x_all.min() - 0.5, 101)
        ax.invert_xaxis()
        ax.set_ylim(bottom=1.0)
        ax.legend(loc="upper left", framealpha=0.95, edgecolor="#cccccc", fontsize=10)

        fig.tight_layout(pad=1.0)
        fig.savefig(panel_paths[0], **SAVE_KW)
        plt.close(fig)
        print(f"Saved: {panel_paths[0]}")

        # ══════════════════════════════════════════════════════════════════════
        # Figure 2 — FOV (degrees) vs Speedup
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(W169, H169))
        setup_academic_axes(ax, grid=False)

        for lr, color, label in LR_CONFIGS:
            c = curves[lr]
            ax.plot(c["fov"], c["fov_speedup"], color=c["color"], linewidth=2.0,
                    label=label, zorder=3)

        ax.set_xlabel("Full FoV (degrees)")
        ax.set_ylabel("Speedup (vs. full resolution)")
        ax.set_title(f"Speedup vs. full FoV\n(fovea = {fov_deg:.0f}°)")
        ax.set_xlim(fov_deg * 0.9, fov_max)
        ax.set_ylim(bottom=1.0)
        ax.legend(loc="upper left", framealpha=0.95, edgecolor="#cccccc", fontsize=10)

        fig.tight_layout(pad=1.0)
        fig.savefig(panel_paths[1], **SAVE_KW)
        plt.close(fig)
        print(f"Saved: {panel_paths[1]}")

        # ══════════════════════════════════════════════════════════════════════
        # Figure 3 — FOV (degrees) vs Token %
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(W169, H169))
        setup_academic_axes(ax, grid=False)

        for lr, color, label in LR_CONFIGS:
            c = curves[lr]
            ax.plot(c["fov"], c["fov_token_pct"], color=c["color"], linewidth=2.0,
                    label=label, zorder=3)

        ax.set_xlabel("Full FoV (degrees)")
        ax.set_ylabel("Tokens (% of original)")
        ax.set_title(f"Token count vs. full FoV\n(fovea = {fov_deg:.0f}°)")
        ax.set_xlim(fov_deg * 0.9, fov_max)
        ax.set_ylim(bottom=0.0, top=105)
        ax.legend(loc="upper right", framealpha=0.95, edgecolor="#cccccc", fontsize=10)

        fig.tight_layout(pad=1.0)
        fig.savefig(panel_paths[2], **SAVE_KW)
        plt.close(fig)
        print(f"Saved: {panel_paths[2]}")


def _downsample_by_factor(arr, factor):
    """Downsample by factor using skimage resize."""
    from skimage.transform import resize

    if factor <= 1:
        return arr
    h, w = arr.shape[0], arr.shape[1]
    out_shape = (h // factor, w // factor) + arr.shape[2:]
    out = resize(arr, out_shape, order=1, anti_aliasing=True, preserve_range=True)
    if arr.dtype == np.uint8:
        out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out


def _center_crop_to_aspect_ratio(image, aspect_ratio_str):
    """
    Center-crop an image so the result has the given aspect ratio (width:height).
    aspect_ratio_str: e.g. "16:9", "1:1", "4:3". If None or empty, return image unchanged.
    image: numpy array (H, W) or (H, W, C).
    """
    if not aspect_ratio_str or not str(aspect_ratio_str).strip():
        return image
    image = np.asarray(image)
    s = str(aspect_ratio_str).strip()
    if ":" not in s:
        return image
    parts = s.split(":", 1)
    try:
        w_ratio = float(parts[0])
        h_ratio = float(parts[1])
    except ValueError:
        return image
    if w_ratio <= 0 or h_ratio <= 0:
        return image
    target_aspect = w_ratio / h_ratio
    h, w = image.shape[0], image.shape[1]
    if w / h >= target_aspect:
        crop_h = h
        crop_w = int(round(h * target_aspect))
    else:
        crop_w = w
        crop_h = int(round(w / target_aspect))
    crop_w = max(1, min(crop_w, w))
    crop_h = max(1, min(crop_h, h))
    y = (h - crop_h) // 2
    x = (w - crop_w) // 2
    return image[y : y + crop_h, x : x + crop_w].copy()


def stack_images(
    images,
    output_path,
    axis="horizontal",
    pad_px=10,
):
    """
    Stack a list of images vertically or horizontally with padding and save.

    Args:
        images: List of numpy arrays (H, W) or (H, W, C). 2D arrays are treated as grayscale (single channel).
        output_path: Path to save the stacked image (e.g. .png).
        axis: "horizontal" or "h" to stack left-to-right; "vertical" or "v" to stack top-to-bottom.
        pad_px: Padding in pixels between adjacent images (default 10).

    Returns:
        The stacked image array (before saving).
    """
    import imageio

    if not images:
        raise ValueError("images must be a non-empty list")

    # Normalize to (H, W, C) and common dtype
    normalized = []
    for img in images:
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        normalized.append(img)

    dtype = normalized[0].dtype
    C = max(im.shape[2] for im in normalized)

    # Ensure all have same number of channels (pad grayscale to 3 if needed for save)
    for i in range(len(normalized)):
        if normalized[i].shape[2] != C:
            normalized[i] = np.repeat(normalized[i], C, axis=2)

    axis = axis.lower()
    if axis in ("horizontal", "h"):
        # Stack horizontally: same height, pad_px between images
        max_h = max(im.shape[0] for im in normalized)
        parts = []
        for k, im in enumerate(normalized):
            if k > 0:
                parts.append(np.full((max_h, pad_px, C), 255, dtype=dtype))
            if im.shape[0] < max_h:
                pad = np.full((max_h - im.shape[0], im.shape[1], C), 255, dtype=dtype)
                im = np.concatenate([im, pad], axis=0)
            parts.append(im)
        stacked = np.concatenate(parts, axis=1)
    elif axis in ("vertical", "v"):
        # Stack vertically: same width, pad_px between images
        max_w = max(im.shape[1] for im in normalized)
        parts = []
        for k, im in enumerate(normalized):
            if k > 0:
                parts.append(np.full((pad_px, max_w, C), 255, dtype=dtype))
            if im.shape[1] < max_w:
                pad = np.full((im.shape[0], max_w - im.shape[1], C), 255, dtype=dtype)
                im = np.concatenate([im, pad], axis=1)
            parts.append(im)
        stacked = np.concatenate(parts, axis=0)
    else:
        raise ValueError(f"axis must be 'horizontal'/'h' or 'vertical'/'v', got {axis!r}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    imageio.imwrite(output_path, stacked)
    print(f"Saved: {output_path}")
    return stacked


def stack_images_grid(images, output_path, num_rows, num_cols, pad_px=10, pad_px_horizontal=None, pad_px_vertical=None):
    """
    Stack a list of images into a num_rows x num_cols grid with padding and save.

    Args:
        images: List of numpy arrays (H, W) or (H, W, C). Length must equal num_rows * num_cols.
        output_path: Path to save the stacked image (e.g. .png).
        num_rows: Number of rows in the grid.
        num_cols: Number of columns in the grid.
        pad_px: Default padding in pixels (used for both directions when pad_px_* are None).
        pad_px_horizontal: Padding between columns; if None, use pad_px.
        pad_px_vertical: Padding between rows; if None, use pad_px.

    Returns:
        The stacked image array (before saving).
    """
    import imageio

    pad_h = pad_px if pad_px_horizontal is None else pad_px_horizontal
    pad_v = pad_px if pad_px_vertical is None else pad_px_vertical

    n = num_rows * num_cols
    if len(images) != n:
        raise ValueError(
            f"Number of images ({len(images)}) must equal num_rows * num_cols ({num_rows} * {num_cols} = {n})"
        )

    if not images:
        raise ValueError("images must be a non-empty list")

    # Normalize to (H, W, C) and common dtype
    normalized = []
    for img in images:
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        normalized.append(img)

    dtype = normalized[0].dtype
    C = max(im.shape[2] for im in normalized)
    for i in range(len(normalized)):
        if normalized[i].shape[2] != C:
            normalized[i] = np.repeat(normalized[i], C, axis=2)

    # Build each row: horizontal stack of num_cols images
    row_images = []
    for r in range(num_rows):
        row_list = normalized[r * num_cols : (r + 1) * num_cols]
        max_h = max(im.shape[0] for im in row_list)
        parts = []
        for k, im in enumerate(row_list):
            if k > 0 and pad_h > 0:
                parts.append(np.full((max_h, pad_h, C), 255, dtype=dtype))
            if im.shape[0] < max_h:
                pad = np.full((max_h - im.shape[0], im.shape[1], C), 255, dtype=dtype)
                im = np.concatenate([im, pad], axis=0)
            parts.append(im)
        row_images.append(np.concatenate(parts, axis=1))

    # Vertical stack of rows (same width per row, pad if needed)
    max_w = max(im.shape[1] for im in row_images)
    parts = []
    for k, im in enumerate(row_images):
        if k > 0 and pad_v > 0:
            parts.append(np.full((pad_v, max_w, C), 255, dtype=dtype))
        if im.shape[1] < max_w:
            pad = np.full((im.shape[0], max_w - im.shape[1], C), 255, dtype=dtype)
            im = np.concatenate([im, pad], axis=1)
        parts.append(im)
    stacked = np.concatenate(parts, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    imageio.imwrite(output_path, stacked)
    print(f"Saved: {output_path}")
    return stacked


def stack_folders(
    input_folder,
    output_folder,
    image_glob="*.png",
    pad_px=10,
    center_crop_aspect=None,
):
    """
    For each subfolder of input_folder, horizontally stack its images into one image.

    - Input: input_folder contains subfolders (e.g. 000, 001, 002, ...).
    - For each subfolder, load images matching image_glob, sorted by filename.
    - Horizontally stack those images (with optional padding) and save to output_folder
      using the subfolder name, e.g. output_folder/000.png.

    Args:
        input_folder: Parent directory whose immediate subdirectories contain images.
        output_folder: Directory to save stacked images (created if needed).
        image_glob: Glob pattern for images in each subfolder (default "*.png").
        pad_px: Horizontal padding between images (passed to stack_images; default 10).

    Returns:
        List of output image paths.
    """
    import glob
    import imageio

    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    subdirs = sorted(
        d for d in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, d))
    )
    if not subdirs:
        raise FileNotFoundError(f"No subfolders found in {input_folder}")

    outputs = []
    for sub in subdirs:
        sub_path = os.path.join(input_folder, sub)
        pattern = os.path.join(sub_path, image_glob)
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"Warning: no images matching {image_glob!r} in {sub_path}, skipping.")
            continue
        images = [imageio.imread(p) for p in paths]
        if center_crop_aspect:
            images = [_center_crop_to_aspect_ratio(img, center_crop_aspect) for img in images]
        out_path = os.path.join(output_folder, f"{sub}.png")
        stack_images(images, output_path=out_path, axis="horizontal", pad_px=pad_px)
        outputs.append(out_path)
    return outputs


def folder_to_video(
    folder_path,
    output_path=None,
    fps=10,
    image_glob="*.png",
    center_crop_aspect=None,
):
    """
    Load images from a folder (sorted by filename), merge into a video, and save in the same folder.

    Args:
        folder_path: Directory containing image files.
        output_path: Output video path. If None, saves to folder_path/video.mp4.
        fps: Frames per second (default 10).
        image_glob: Glob pattern for image files (default "*.png"), e.g. "*.jpg" or "img_*.png".

    Returns:
        The path to the saved video file.
    """
    import glob
    import imageio

    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pattern = os.path.join(folder_path, image_glob)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No images matching {image_glob!r} in {folder_path}")

    if output_path is None:
        output_path = os.path.join(folder_path, "video.mp4")
    else:
        output_path = os.path.abspath(output_path)

    images = []
    for p in paths:
        img = imageio.imread(p)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        img = _center_crop_to_aspect_ratio(img, center_crop_aspect)
        images.append(img)

    with imageio.get_writer(output_path, fps=fps) as writer:
        for img in images:
            writer.append_data(img)

    print(f"Saved: {output_path} ({len(images)} frames @ {fps} fps)")
    return output_path


def _detect_white_circle(frame):
    """
    Detect a white circular border (or bright circular region) in a frame.
    Returns (cx, cy, radius_px) or None if no suitable circle found.
    Uses bright pixel threshold, contours, and minEnclosingCircle (same radius, arbitrary location).
    """
    try:
        import cv2
    except ImportError:
        return None
    if frame.ndim == 3:
        gray = np.mean(frame.astype(np.float32), axis=2).astype(np.uint8)
    else:
        gray = frame.astype(np.uint8)
    h, w = gray.shape
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Take contour with largest area (main white region)
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None
    (cx, cy), radius_px = cv2.minEnclosingCircle(largest)
    cx, cy, radius_px = float(cx), float(cy), float(radius_px)
    min_r = 0.05 * min(h, w)
    max_r = 0.95 * min(h, w) / 2
    if radius_px < min_r or radius_px > max_r:
        return None
    return (cx, cy, radius_px)


def _draw_circle_overlay(frame, cx, cy, radius_px, outline_width_frac, color):
    """
    Draw a circle overlay on frame (in-place). Same radius as detected; thickness and color from args.
    outline_width_frac: line width as fraction of image width (reuse foveation outline convention).
    color: (R, G, B) tuple.
    """
    try:
        import cv2
    except ImportError:
        return
    h, w = frame.shape[0], frame.shape[1]
    thickness = max(1, int(outline_width_frac * w))
    # cv2.circle expects BGR when image is BGR; imageio frames are typically RGB
    color_bgr = (int(color[2]), int(color[1]), int(color[0]))
    cv2.circle(frame, (int(round(cx)), int(round(cy))), int(round(radius_px)), color_bgr, thickness)

# for video
def draw_foveation_circle_from_params(video_frames, keyframe_centers, keyframe_radii, outline_color=(255, 255, 255)):
    """Draw a white circle on each frame using smoothly interpolated keyframe parameters.
    Uses the same normalized-to-pixel conversion as create_foveation_mask."""
    import cv2
    num_frames = len(video_frames)
    n_keys = len(keyframe_centers)
    keyframe_t = np.linspace(0, 1, n_keys)
    frame_t = np.linspace(0, 1, num_frames)

    cx_interp = np.interp(frame_t, keyframe_t, [c[0] for c in keyframe_centers])
    cy_interp = np.interp(frame_t, keyframe_t, [c[1] for c in keyframe_centers])
    r_interp = np.interp(frame_t, keyframe_t, keyframe_radii)

    out = []
    for t, frame in enumerate(video_frames):
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        elif hasattr(frame, "__array__"):
            frame = np.array(frame)
        else:
            frame = np.array(frame)
        if frame.dtype != np.uint8 or frame.max() <= 1.0:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        h, w = frame.shape[:2]
        px_cx = int((cx_interp[t] + 0.5) * w)
        px_cy = int((cy_interp[t] + 0.5) * h)
        diagonal = (h**2 + w**2) ** 0.5
        px_r = int(round(r_interp[t] * (diagonal / 2.0)))
        frame = cv2.circle(frame, (px_cx, px_cy), px_r, outline_color, 2)
        out.append(frame)
    return out


def _extract_equally_spaced_frames(
    video_path,
    n,
    downsample_factor=None,
    draw_circle_overlay=False,
    outline_width_frac=0.005,
    outline_color=(255, 0, 0),
    center_crop_aspect=None,
):
    """
    Helper: extract n frames from a video at equally spaced positions.
    Returns a list of frames as numpy arrays.
    If draw_circle_overlay is True, detects a white circular border in each frame and overlays
    a circle of the same radius with the given thickness (outline_width_frac) and color.
    If center_crop_aspect is set (e.g. "16:9"), each frame is center-cropped to that aspect ratio.
    """
    import imageio

    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = imageio.get_reader(video_path)
    # Robustly determine number of frames; guard against backends that report inf or 0.
    try:
        try:
            num_frames = reader.get_length()
        except Exception:
            num_frames = len(reader)
    except Exception:
        num_frames = np.inf

    if not np.isfinite(num_frames) or int(num_frames) <= 0:
        # Fallback: iterate once to count frames, then reopen reader for actual sampling.
        count = 0
        try:
            for _ in reader:
                count += 1
        finally:
            reader.close()
        num_frames = count
        reader = imageio.get_reader(video_path)
    else:
        num_frames = int(num_frames)

    if num_frames < n:
        reader.close()
        raise ValueError(
            f"Video has only {num_frames} frames, but requested {n} frames. "
            "Cannot sample enough frames."
        )

    if isinstance(outline_color, (list, tuple)) and len(outline_color) == 3:
        color = tuple(int(x) for x in outline_color)
    else:
        color = (255, 0, 0)

    indices = np.linspace(0, num_frames - 1, n, dtype=int)
    frames = []
    for idx in indices:
        frame = reader.get_data(int(idx))
        if frame.ndim == 2:
            frame = np.expand_dims(frame, axis=-1)
        if downsample_factor is not None and downsample_factor > 1:
            frame = _downsample_by_factor(frame, downsample_factor)
        frame = _center_crop_to_aspect_ratio(frame, center_crop_aspect)
        if draw_circle_overlay:
            circle = _detect_white_circle(frame)
            if circle is not None:
                cx, cy, r = circle
                _draw_circle_overlay(frame, cx, cy, r, outline_width_frac, color)
        frames.append(frame)
    reader.close()
    return frames


def generate_video_frame_grid(
    video_path,
    output_path,
    num_rows,
    num_cols,
    pad_px=10,
    downsample_factor=None,
    draw_circle_overlay=False,
    outline_width_frac=0.005,
    outline_color=(255, 0, 0),
    center_crop_aspect=None,
):
    """
    Extract N = num_rows * num_cols frames from a video at equally spaced
    positions, and stack them into a num_rows x num_cols grid with padding.

    Args:
        video_path: Path to input video file.
        output_path: Path to save the stacked grid image (e.g. .png).
        num_rows: Number of rows in the grid.
        num_cols: Number of columns in the grid.
        pad_px: Padding in pixels between frames in the grid.
        downsample_factor: Optional integer; if >1, each frame is downsampled
            by this factor using _downsample_by_factor (same helper as other grids).
        draw_circle_overlay: If True, detect white circular border in each frame and overlay a circle (same radius, different thickness/color).
        outline_width_frac: Line width as fraction of image width (foveation outline convention).
        outline_color: (R, G, B) for overlay circle.
        center_crop_aspect: If set (e.g. "16:9"), center-crop each frame to that aspect ratio.
    """
    n = num_rows * num_cols
    frames = _extract_equally_spaced_frames(
        video_path,
        n,
        downsample_factor=downsample_factor,
        draw_circle_overlay=draw_circle_overlay,
        outline_width_frac=outline_width_frac,
        outline_color=outline_color,
        center_crop_aspect=center_crop_aspect,
    )
    print(f"Sampling {n} frames from video: {video_path}")
    stack_images_grid(frames, output_path=output_path, num_rows=num_rows, num_cols=num_cols, pad_px=pad_px)
    return output_path


def generate_baseline_comparison_grid(
    base_dir,
    baselines,
    total_images,
    num_cols,
    output_path,
    padding_frac=0.02,
    downsample_factor=None,
    image_name_template="img_{:010d}.png",
    num_rows=None,
    image_ids=None,
    center_crop_aspect=None,
):
    """
    Stack images from multiple baseline folders into an interleaved grid.

    - Each image is downsampled by downsample_factor (default num_cols) first.
    - Then stacking is performed on the downsampled images (padding = padding_frac * downsampled image width).
    - Reads images from base_dir / baseline / image_name_template for each baseline.
    - Each row has num_cols images from one baseline; rows are interleaved by baseline.
    - total_images must be divisible by num_cols (when num_rows is None).
    - When num_rows is not None:
        * We form 2 * num_rows prompt indices.
        * By default, these indices are sampled in a fixed, deterministic way from [0, total_images).
        * If image_ids is provided, those indices are used instead (and must have length 2 * num_rows).

    Returns the final image array and saves to output_path.
    """
    import imageio

    factor = downsample_factor if downsample_factor is not None else num_cols

    # Load one image to get dimensions, then get downsampled size
    sample_path = os.path.join(base_dir, baselines[0], image_name_template.format(0))
    if not os.path.isfile(sample_path):
        sample_path = os.path.join(base_dir, baselines[0], image_name_template.format(1))
    sample = imageio.imread(sample_path)
    if sample.ndim == 2:
        sample = np.expand_dims(sample, axis=-1)
    sample = _center_crop_to_aspect_ratio(sample, center_crop_aspect)
    if factor > 1:
        sample_small = _downsample_by_factor(sample, factor)
    else:
        sample_small = sample
    H_small, W_small = sample_small.shape[0], sample_small.shape[1]
    C = sample_small.shape[2]
    pad_px = int(padding_frac * W_small)

    # If num_rows is not specified, keep the original behavior for backwards compatibility:
    # each row has num_cols images from a single baseline, rows are interleaved by baseline.
    if num_rows is None:
        if total_images % num_cols != 0:
            raise ValueError(f"total_images ({total_images}) must be divisible by num_cols ({num_cols})")
        num_baselines = len(baselines)
        num_row_groups = total_images // num_cols
        total_rows = num_row_groups * num_baselines

        row_images = []
        for r in range(total_rows):
            baseline_idx = r % num_baselines
            group = r // num_baselines
            start = group * num_cols
            end = start + num_cols
            folder = os.path.join(base_dir, baselines[baseline_idx])
            parts = []
            for i in range(start, end):
                path = os.path.join(folder, image_name_template.format(i))
                img = imageio.imread(path)
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
                img = _center_crop_to_aspect_ratio(img, center_crop_aspect)
                if factor > 1:
                    img = _downsample_by_factor(img, factor)
                parts.append(img)
            # Horizontal stack with padding between images
            row_parts = [parts[0]]
            for p in parts[1:]:
                row_parts.append(np.full((H_small, pad_px, C), 255, dtype=parts[0].dtype))
                row_parts.append(p)
            row_img = np.concatenate(row_parts, axis=1)
            row_images.append(row_img)

        # Vertical stack with padding between rows
        row_w = row_images[0].shape[1]
        pad_row = np.full((pad_px, row_w, C), 255, dtype=row_images[0].dtype)
        grid_parts = [row_images[0]]
        for r in row_images[1:]:
            grid_parts.append(pad_row)
            grid_parts.append(r)
        grid = np.concatenate(grid_parts, axis=0)
    else:
        # New layout: each prompt is a horizontal baseline triplet; we place two triplets per row
        # (left and right), so the final grid has N rows and 6 columns in total when len(baselines) == 3.
        num_prompts = 2 * num_rows

        # Determine which global image indices to use for each prompt slot.
        if image_ids is not None:
            if len(image_ids) != num_prompts:
                raise ValueError(
                    f"image_ids must have length 2 * num_rows ({num_prompts}), "
                    f"got {len(image_ids)}."
                )
            indices = list(image_ids)
        else:
            if total_images < num_prompts:
                raise ValueError(
                    f"total_images ({total_images}) must be at least 2 * num_rows ({num_prompts}) "
                    "to create two prompt triplets per row."
                )
            # Fixed, deterministic sampling of indices across [0, total_images).
            # This provides a stable subset of examples for the paper figures.
            if total_images == num_prompts:
                indices = list(range(num_prompts))
            else:
                indices = np.linspace(0, total_images - 1, num_prompts, dtype=int).tolist()

        # Pre-load all triplets (one per chosen prompt index)
        triplets = []
        for idx in indices:
            imgs = []
            for baseline in baselines:
                folder = os.path.join(base_dir, baseline)
                path = os.path.join(folder, image_name_template.format(idx))
                img = imageio.imread(path)
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
                img = _center_crop_to_aspect_ratio(img, center_crop_aspect)
                if factor > 1:
                    img = _downsample_by_factor(img, factor)
                imgs.append(img)
            # Stack baselines horizontally to form a single prompt triplet
            triplet_parts = [imgs[0]]
            for img in imgs[1:]:
                triplet_parts.append(np.full((H_small, pad_px, C), 255, dtype=imgs[0].dtype))
                triplet_parts.append(img)
            triplet_img = np.concatenate(triplet_parts, axis=1)
            triplets.append(triplet_img)

        # Build rows: two prompt triplets per row, stacked horizontally
        row_images = []
        for r in range(num_rows):
            left = triplets[2 * r]
            right = triplets[2 * r + 1]
            # Make sure widths match by padding if necessary
            max_w = max(left.shape[1], right.shape[1])
            if left.shape[1] < max_w:
                left = np.concatenate(
                    [left, np.full((left.shape[0], max_w - left.shape[1], C), 255, dtype=left.dtype)],
                    axis=1,
                )
            if right.shape[1] < max_w:
                right = np.concatenate(
                    [right, np.full((right.shape[0], max_w - right.shape[1], C), 255, dtype=right.dtype)],
                    axis=1,
                )
            col_pad = np.full((H_small, pad_px, C), 255, dtype=left.dtype)
            row_img = np.concatenate([left, col_pad, right], axis=1)
            row_images.append(row_img)

        # Vertical stack with padding between rows
        row_w = row_images[0].shape[1]
        pad_row = np.full((pad_px, row_w, C), 255, dtype=row_images[0].dtype)
        grid_parts = [row_images[0]]
        for r in row_images[1:]:
            grid_parts.append(pad_row)
            grid_parts.append(r)
        grid = np.concatenate(grid_parts, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    imageio.imwrite(output_path, grid)
    print(f"Saved: {output_path}")
    return grid


def generate_baseline_video_grid(
    base_dir,
    baselines,
    video_ids,
    num_rows,
    num_cols,
    pad_px,
    output_path,
    downsample_factor=None,
    pad_px_horizontal=None,
    pad_px_vertical=None,
    draw_circle_overlay=False,
    outline_width_frac=0.005,
    outline_color=(255, 0, 0),
    center_crop_aspect=None,
):
    """
    Video analogue of generate_baseline_comparison_grid with flexible number of videos.

    New layout:
    - For each baseline and video ID, loads base_dir / baseline / {id}.mp4.
    - For each video, extracts num_cols frames uniformly spaced in time.
    - For a fixed video ID, frames are stacked horizontally (columns = num_cols) and
      baselines are stacked vertically (rows = num_baselines), forming a grid of
      shape [num_baselines x num_cols].
    - These per-video grids are then stacked vertically in the order of video_ids,
      so the final grid has:
        rows  = len(baselines) * len(video_ids)
        cols  = num_cols
        total = len(video_ids) * num_cols * len(baselines) images.

    Padding: pad_px is the default; pad_px_horizontal/pad_px_vertical override for grid spacing
    (e.g. pad_px_horizontal=0 to remove horizontal padding between images).
    Foveation circles are drawn from center_list/radius_list (spline_prompts.csv) via
    draw_foveation_circle_from_params; draw_circle_overlay is not used.
    """
    import imageio  # only to ensure dependency present; frame extraction uses helper

    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    num_baselines = len(baselines)
    num_videos = len(video_ids)
    if num_baselines == 0:
        raise ValueError("generate_baseline_video_grid expects at least one baseline")
    if num_videos == 0:
        raise ValueError("generate_baseline_video_grid expects at least one video ID")

    frames_per_video = num_cols
    total_rows = num_baselines * num_videos
    if total_rows <= 0:
        raise ValueError("Total rows for baseline video grid must be positive")

    # Load keyframe centers and radii per video from spline_prompts (by id)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    spline_path = os.path.join(script_dir, "prompts", "spline_prompts.csv")
    spline_prompts = pd.read_csv(spline_path)
    # Resolve center_list and radius_list per video (parse CSV strings)
    keyframe_centers_per_video = []
    keyframe_radii_per_video = []
    for vid_id in video_ids:
        row = spline_prompts[spline_prompts["id"] == int(vid_id)].iloc[0]
        keyframe_centers_per_video.append(ast.literal_eval(row["center_list"]))
        keyframe_radii_per_video.append(ast.literal_eval(row["radius_list"]))

    # Extract frames per baseline and video ID (no circle overlay in extract)
    frames_bv = []  # shape: [baseline][video][frames_per_video]
    for baseline in baselines:
        baseline_frames = []
        for vid_id in video_ids:
            vid_name = f"{vid_id}.mp4"
            video_path = os.path.join(base_dir, baseline, vid_name)
            frames = _extract_equally_spaced_frames(
                video_path,
                frames_per_video,
                downsample_factor=downsample_factor,
                draw_circle_overlay=False,
                outline_width_frac=outline_width_frac,
                outline_color=outline_color,
                center_crop_aspect=center_crop_aspect,
            )
            baseline_frames.append(frames)
        frames_bv.append(baseline_frames)

    # Draw foveation circle on each baseline×video frame list from params
    if draw_circle_overlay:
        for b in range(num_baselines):
            if b > 0: # not high-res
                for v in range(num_videos):
                    frames_bv[b][v] = draw_foveation_circle_from_params(
                        frames_bv[b][v],
                        keyframe_centers_per_video[v],
                        keyframe_radii_per_video[v],
                        outline_color=outline_color,
                    )

    # Build images list in row-major order for stack_images_grid:
    # For each video (in video_ids order), for each baseline (in baselines order),
    # append its frames left-to-right. This yields:
    #   rows = num_videos * num_baselines
    #   cols = frames_per_video (= num_cols)
    images = []
    for v in range(num_videos):
        for b in range(num_baselines):
            frames = frames_bv[b][v]
            if len(frames) != frames_per_video:
                raise ValueError(
                    f"Expected {frames_per_video} frames for baseline '{baselines[b]}' "
                    f"video '{video_ids[v]}', got {len(frames)}"
                )
            images.extend(frames)

    stacked = stack_images_grid(
        images,
        output_path=output_path,
        num_rows=total_rows,
        num_cols=num_cols,
        pad_px=pad_px,
        pad_px_horizontal=pad_px_horizontal,
        pad_px_vertical=pad_px_vertical,
    )
    return stacked

def generate_foveation_trajectory_figure(
    base_dir,
    row1_name,
    row2_name,
    subfolder_indices=0,
    output_path=None,
    padding_frac=0.02,
    downsample_factor=None,
    num_rows=None,
    image_name_glob="img_*.png",
    center_crop_aspect=None,
):
    """
    Generate a figure showing foveation trajectory sequences.
    - subfolder_indices: single int or list of ints (e.g. 0 or [0, 1, 2] for 000, 001, 002).
    - For each subfolder index: row 1 from base_dir/row1_name/{subfolder}/; row 2 from base_dir/row2_name/{subfolder}/
      if row2_name is provided. If row2_name is None or empty, only row1 is plotted.
    - Default behavior (num_rows is None): all img_*.png in each subfolder are stacked horizontally; rows stacked
      vertically with the same padding between rows.
    - New behavior (num_rows is not None): columns are foveation trajectories. If row2_name is set, row1 and row2
      are interleaved (row1, row2, ...); otherwise only row1 columns are shown.
    """
    import glob
    import imageio

    if isinstance(subfolder_indices, int):
        subfolder_indices = [subfolder_indices]
    has_row2 = row2_name is not None and (
        not isinstance(row2_name, str) or (row2_name and row2_name.strip())
    )
    factor = downsample_factor

    def load_row_images(folder, factor):
        paths = sorted(glob.glob(os.path.join(folder, image_name_glob)))
        if not paths:
            raise FileNotFoundError(f"No images matching {image_name_glob} in {folder}")
        imgs = []
        for p in paths:
            img = imageio.imread(p)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            img = _center_crop_to_aspect_ratio(img, center_crop_aspect)
            if factor and factor > 1:
                img = _downsample_by_factor(img, factor)
            imgs.append(img)
        return imgs

    def horizontal_stack(imgs, H, W, pad_px, C, dtype):
        parts = [imgs[0]]
        for p in imgs[1:]:
            parts.append(np.full((H, pad_px, C), 255, dtype=dtype))
            parts.append(p)
        return np.concatenate(parts, axis=1)

    # First pass: load one image to get H, W, C and pad_px
    subfolder = f"{subfolder_indices[0]:03d}"
    dir1 = os.path.join(base_dir, row1_name, subfolder)
    if not os.path.isdir(dir1):
        raise FileNotFoundError(f"Folder not found: {dir1}")
    sample_imgs = load_row_images(dir1, factor)
    H, W = sample_imgs[0].shape[0], sample_imgs[0].shape[1]
    C = sample_imgs[0].shape[2]
    pad_px = int(padding_frac * W)
    dtype = sample_imgs[0].dtype
    print(pad_px)

    if num_rows is None:
        # Original behavior: horizontal trajectories stacked as rows.
        row_blocks = []
        row_w_max = 0
        for idx in subfolder_indices:
            subfolder = f"{idx:03d}"
            dir1 = os.path.join(base_dir, row1_name, subfolder)
            if not os.path.isdir(dir1):
                raise FileNotFoundError(f"Folder not found: {dir1}")
            row1_imgs = load_row_images(dir1, factor)
            row1 = horizontal_stack(row1_imgs, H, W, pad_px, C, dtype)
            row_w_max = max(row_w_max, row1.shape[1])
            row2 = None
            if has_row2:
                dir2 = os.path.join(base_dir, row2_name, subfolder)
                if not os.path.isdir(dir2):
                    raise FileNotFoundError(f"Folder not found: {dir2}")
                row2_imgs = load_row_images(dir2, factor)
                row2 = horizontal_stack(row2_imgs, H, W, pad_px, C, dtype)
                row_w_max = max(row_w_max, row2.shape[1])
            row_blocks.append((row1, row2))

        # Pad rows to same width and stack vertically with padding between rows
        pad_row_template = None
        grid_parts = []
        for row1, row2 in row_blocks:
            if row1.shape[1] < row_w_max:
                row1 = np.concatenate(
                    [row1, np.full((row1.shape[0], row_w_max - row1.shape[1], C), 255, dtype=dtype)],
                    axis=1,
                )
            if pad_row_template is None:
                pad_row_template = np.full((pad_px, row_w_max, C), 255, dtype=dtype)
            grid_parts.append(row1)
            if has_row2 and row2 is not None:
                if row2.shape[1] < row_w_max:
                    row2 = np.concatenate(
                        [row2, np.full((row2.shape[0], row_w_max - row2.shape[1], C), 255, dtype=dtype)],
                        axis=1,
                    )
                grid_parts.append(pad_row_template)
                grid_parts.append(row2)
                grid_parts.append(pad_row_template)
        if grid_parts:
            grid_parts = grid_parts[:-1]  # drop trailing padding between blocks
        grid = np.concatenate(grid_parts, axis=0) if grid_parts else np.zeros((0, 0, C), dtype=dtype)
    else:
        # New behavior: each column is a trajectory; optionally interleave row1/row2 columns.
        use_indices = subfolder_indices[:1]

        columns = []

        def vertical_stack(frames):
            col_w = frames[0].shape[1]
            parts = [frames[0]]
            for f in frames[1:]:
                parts.append(np.full((pad_px, col_w, C), 255, dtype=dtype))
                parts.append(f)
            return np.concatenate(parts, axis=0)

        for idx in use_indices:
            subfolder = f"{idx:03d}"
            dir1 = os.path.join(base_dir, row1_name, subfolder)
            if not os.path.isdir(dir1):
                raise FileNotFoundError(f"Folder not found: {dir1}")

            row1_imgs = load_row_images(dir1, factor)
            num_frames = len(row1_imgs)
            if num_frames < num_rows:
                raise ValueError(
                    f"Not enough frames in {dir1} for num_rows={num_rows}: only {num_frames} frames."
                )

            if num_frames == num_rows:
                frame_indices = list(range(num_rows))
            else:
                frame_indices = np.linspace(0, num_frames - 1, num_rows, dtype=int)
                if "radius" in base_dir:
                    frame_indices = list(range(num_rows))

            col1_frames = [row1_imgs[i] for i in frame_indices]
            col1 = vertical_stack(col1_frames)
            columns.append(col1)

            if has_row2:
                dir2 = os.path.join(base_dir, row2_name, subfolder)
                if not os.path.isdir(dir2):
                    raise FileNotFoundError(f"Folder not found: {dir2}")
                row2_imgs = load_row_images(dir2, factor)
                if len(row1_imgs) != len(row2_imgs):
                    raise ValueError(
                        f"Mismatch in number of frames between {dir1} ({len(row1_imgs)}) and "
                        f"{dir2} ({len(row2_imgs)})."
                    )
                col2_frames = [row2_imgs[i] for i in frame_indices]
                col2 = vertical_stack(col2_frames)
                columns.append(col2)

        # Pad columns to same height and stack horizontally with padding between columns
        if columns:
            max_h = max(col.shape[0] for col in columns)
            padded_columns = []
            for col in columns:
                if col.shape[0] < max_h:
                    pad = np.full((max_h - col.shape[0], col.shape[1], C), 255, dtype=dtype)
                    col = np.concatenate([col, pad], axis=0)
                padded_columns.append(col)

            grid_parts = [padded_columns[0]]
            for col in padded_columns[1:]:
                grid_parts.append(np.full((max_h, pad_px, C), 255, dtype=dtype))
                grid_parts.append(col)
            grid = np.concatenate(grid_parts, axis=1)
        else:
            grid = np.zeros((0, 0, C), dtype=dtype)

    if output_path is None:
        output_path = os.path.join(base_dir, "foveation_trajectory_figure.png")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    imageio.imwrite(output_path, grid)
    print(f"Saved: {output_path}")
    return grid


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "figure",
        type=str,
        choices=[
            "runtime",
            "runtime_quality",
            "runtime_joint",
            "baselines",
            "baselines_video",
            "foveation_trajectory",
            "stack",
            "video",
            "video_grid",
            "stack_folders",
        ],
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for the figure")
    parser.add_argument("--base-dir", type=str, default=None, help="Base dir (default: ../results/eccv2026_final_v2)")
    parser.add_argument("--fov-max", type=float, default=180.0, help="Maximum full FoV (degrees) shown in FOV panels of runtime_joint figure (default: 180)")
    parser.add_argument("--padding-frac", type=float, default=0.02, help="Padding between images as fraction of image width")
    parser.add_argument("--downsample", type=int, default=None, help="Downsample factor for foveation_trajectory (optional)")
    parser.add_argument(
        "--center-crop-aspect",
        type=str,
        default=None,
        help="Center-crop all loaded images and video frames to this aspect ratio (e.g. '16:9', '1:1'). If not set, no cropping.",
    )

    # baselines / foveation_trajectory figure-specific arguments
    parser.add_argument("--baselines", type=str, nargs="+", default=["baseline_eval", "naive_eval", "ours_dit_eval"], help="Baseline folder names for baselines figure")
    parser.add_argument("--total-images", type=int, default=20, help="Number of images per baseline")
    parser.add_argument("--num-cols", type=int, default=4, help="Number of images per row")
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="Number of rows for baselines, foveation_trajectory, or stack figure (stack: grid is num-rows x num-cols)",
    )

    parser.add_argument(
        "--image-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit image indices to use for baseline figure when num-rows is provided. "
            "Length must be exactly 2 * num-rows (left/right triplets per row)."
        ),
    )

    # stack figure: list of images in a num_rows x num_cols grid
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=None,
        help="Image paths for stack figure (required when figure=stack). Count must equal --num-rows * --num-cols.",
    )
    parser.add_argument("--pad-px", type=int, default=10, help="Padding in pixels between images for stack figure")
    parser.add_argument(
        "--no-horizontal-padding",
        action="store_true",
        help="For baselines_video: use zero horizontal padding between images (vertical padding still uses --pad-px).",
    )
    parser.add_argument(
        "--no-vertical-padding",
        action="store_true",
        help="For baselines_video: use zero vertical padding between rows (horizontal padding still uses --pad-px).",
    )
    # Video frame circle overlay (detect white circular border, overlay same-radius circle; reuse foveation outline convention)
    parser.add_argument(
        "--draw-circle-overlay",
        action="store_true",
        help="For video_grid and baselines_video: detect white circular border in each frame and overlay a circle (same radius, different thickness/color).",
    )
    parser.add_argument(
        "--outline-width-frac",
        type=float,
        default=0.005,
        help="Circle overlay line width as fraction of image width (default 0.005). Used with --draw-circle-overlay.",
    )
    parser.add_argument(
        "--outline-color",
        type=str,
        default="255,0,0",
        help="Circle overlay color as R,G,B (default red: 255,0,0). Used with --draw-circle-overlay.",
    )

    # video / stack_folders figure: folder of images -> video or per-subfolder stack
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder of images (figure=video) or parent folder containing subfolders (figure=stack_folders)",
    )
    parser.add_argument("--fps", type=float, default=10, help="Frames per second for output video")
    parser.add_argument("--image-glob", type=str, default="*.png", help="Glob for image files in folder (default: *.png)")

    # foveation_trajectory figure-specific arguments
    parser.add_argument("--row1-name", type=str, default="foveation_trajectory_naive", help="Folder name for row 1 of foveation_trajectory (under base-dir)")
    parser.add_argument("--row2-name", type=str, default="foveation_trajectory_ours_dit", help="Folder name for row 2 of foveation_trajectory; use empty string or 'none' to plot only row1")
    parser.add_argument("--subfolder-idx", type=int, nargs="+", default=[0], help="Subfolder index(s), e.g. 0 or 0 1 2 for 000, 001, 002")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_base = os.path.join(script_dir, "..", "results", "eccv2026_final_v2")

    if args.figure == "runtime":
        generate_runtime_figure(args.base_dir, output_path=args.output)
    elif args.figure == "runtime_quality":
        file_path = os.path.join(args.base_dir, "runtime_and_radius_list.npz")
        generate_runtime_quality_two_col(file_path, output_path=args.output)
    elif args.figure == "runtime_joint":
        base_dir = args.base_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "results", "eccv2026_final_v2", "runtime_apollo_v2",
        )
        generate_runtime_joint_figure(base_dir, output_path=args.output, fov_max=args.fov_max)
    elif args.figure == "baselines":
        base_dir = args.base_dir or default_base
        generate_baseline_comparison_grid(
            base_dir=base_dir,
            baselines=args.baselines,
            total_images=args.total_images,
            num_cols=args.num_cols,
            output_path=args.output or os.path.join(base_dir, "baselines.png"),
            padding_frac=getattr(args, "padding_frac", 0.02),
            num_rows=args.num_rows,
            image_ids=args.image_ids,
            downsample_factor=args.downsample,
            center_crop_aspect=getattr(args, "center_crop_aspect", None),
        )
    elif args.figure == "baselines_video":
        base_dir = args.base_dir or default_base
        if args.image_ids is None or len(args.image_ids) == 0:
            raise ValueError(
                "figure=baselines_video expects at least one --image-ids value "
                "(video IDs corresponding to {id}.mp4)."
            )
        # New layout: num_cols controls the number of frames per video (horizontal),
        # and rows are determined by baselines x number of videos.
        num_rows = len(args.baselines) * len(args.image_ids)
        num_cols = args.num_cols
        pad_px = args.pad_px
        pad_px_horizontal = 0 if getattr(args, "no_horizontal_padding", False) else None
        pad_px_vertical = 0 if getattr(args, "no_vertical_padding", False) else None
        output_path = args.output or os.path.join(base_dir, "baselines_video_grid.png")
        outline_color = (255, 0, 0)
        if getattr(args, "outline_color", None):
            outline_color = tuple(int(x) for x in args.outline_color.split(","))
        generate_baseline_video_grid(
            base_dir=base_dir,
            baselines=args.baselines,
            video_ids=args.image_ids,
            num_rows=num_rows,
            num_cols=num_cols,
            pad_px=pad_px,
            output_path=output_path,
            downsample_factor=args.downsample,
            pad_px_horizontal=pad_px_horizontal,
            pad_px_vertical=pad_px_vertical,
            draw_circle_overlay=getattr(args, "draw_circle_overlay", False),
            outline_width_frac=getattr(args, "outline_width_frac", 0.005),
            outline_color=outline_color,
            center_crop_aspect=getattr(args, "center_crop_aspect", None),
        )
    elif args.figure == "foveation_trajectory":
        base_dir = args.base_dir or default_base
        row2_name = args.row2_name
        if isinstance(row2_name, str) and row2_name.strip().lower() in ("", "none"):
            row2_name = None
        generate_foveation_trajectory_figure(
            base_dir=base_dir,
            row1_name=args.row1_name,
            row2_name=row2_name,
            subfolder_indices=args.subfolder_idx,
            output_path=args.output,
            padding_frac=args.padding_frac,
            downsample_factor=args.downsample,
            num_rows=args.num_rows,
            center_crop_aspect=getattr(args, "center_crop_aspect", None),
        )
    elif args.figure == "stack":
        if not args.images:
            raise ValueError("--images is required for figure=stack (provide one or more image paths)")
        if args.num_rows is None:
            raise ValueError("--num-rows is required for figure=stack (grid layout)")
        num_rows = args.num_rows
        num_cols = args.num_cols
        n = num_rows * num_cols
        if len(args.images) != n:
            raise ValueError(
                f"Number of images ({len(args.images)}) must equal num_rows * num_cols ({num_rows} * {num_cols} = {n})"
            )
        import imageio
        images = [imageio.imread(p) for p in args.images]
        if getattr(args, "center_crop_aspect", None):
            images = [_center_crop_to_aspect_ratio(img, args.center_crop_aspect) for img in images]
        if args.downsample:
            print(f"Downsampling images by factor {args.downsample}")
            images = [_downsample_by_factor(img, args.downsample) for img in images]
        out = args.output or "stacked.png"
        stack_images_grid(
            images,
            output_path=out,
            num_rows=num_rows,
            num_cols=num_cols,
            pad_px=int(args.pad_px // args.downsample),
        )
    elif args.figure == "video":
        if not args.folder:
            raise ValueError("--folder is required for figure=video")
        folder_to_video(
            folder_path=args.folder,
            output_path=args.output,
            fps=args.fps,
            image_glob=args.image_glob,
            center_crop_aspect=getattr(args, "center_crop_aspect", None),
        )
    elif args.figure == "video_grid":
        if not args.folder:
            raise ValueError("--folder is required for figure=video_grid (video path)")
        if args.num_rows is None:
            raise ValueError("--num-rows is required for figure=video_grid (grid layout)")
        num_rows = args.num_rows
        num_cols = args.num_cols
        video_path = args.folder
        out = args.output or os.path.join(os.path.dirname(os.path.abspath(video_path)), "video_frame_grid.png")
        outline_color = (255, 0, 0)
        if getattr(args, "outline_color", None):
            outline_color = tuple(int(x) for x in args.outline_color.split(","))
        generate_video_frame_grid(
            video_path=video_path,
            output_path=out,
            num_rows=num_rows,
            num_cols=num_cols,
            pad_px=args.pad_px,
            downsample_factor=args.downsample,
            draw_circle_overlay=getattr(args, "draw_circle_overlay", False),
            outline_width_frac=getattr(args, "outline_width_frac", 0.005),
            outline_color=outline_color,
            center_crop_aspect=getattr(args, "center_crop_aspect", None),
        )
    elif args.figure == "stack_folders":
        if not args.folder:
            raise ValueError("--folder is required for figure=stack_folders")
        input_folder = args.folder
        output_folder = args.output or os.path.join(input_folder, "stacked")
        stack_folders(
            input_folder=input_folder,
            output_folder=output_folder,
            image_glob=args.image_glob,
            pad_px=args.pad_px,
            center_crop_aspect=getattr(args, "center_crop_aspect", None),
        )
    else:
        raise ValueError(f"Invalid figure: {args.figure}")
