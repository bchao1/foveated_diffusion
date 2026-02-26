"""
Generate publication-quality figures for foveated diffusion paper.
Academic style: faint dashed grids, serif fonts, colorblind-friendly palette.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from argparse import ArgumentParser

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


def generate_runtime_figure(runtime_and_radius_list_path, output_path=None):
    """
    Plot speedup vs. token budget (% of original tokens) from saved npz.
    X = token percentage (ratio = r^2 + (1-r^2)/4), displayed decreasing left-to-right.
    Y = speedup factor vs. full-resolution (r=1) generation (baseline_runtime / runtime).
    Expects npz with keys: runtime_list, r_list.
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

        # X = token percentage (ratio = r^2 + (1-r^2)/4), in %
        token_ratio = r_to_token_percentage(r_list)
        token_pct = token_ratio * 100  # e.g. 25 to 100

        # Y = speedup vs full res (r=1): baseline_runtime / runtime
        baseline_runtime = np.asarray(runtime_list)[np.argmax(token_pct)]  # runtime at full res (max tokens)
        speedup = baseline_runtime / np.asarray(runtime_list)

        # Sort by token_pct increasing so curve is well-defined; then invert x so % decreases left→right
        order = np.argsort(token_pct)
        x = token_pct[order]
        y = speedup[order]

        fig, ax = plt.subplots(figsize=(4, 4))
        setup_academic_axes(ax, grid=True)

        ax.plot(
            x, y,
            color=COLORS["primary"],
            linewidth=2.0,
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="Speedup",
            zorder=3,
        )

        ax.set_xlabel("Tokens (% of original)")
        ax.set_ylabel("Speedup (vs. full resolution)")
        ax.set_title("Inference speedup vs. token budget")
        ax.set_xlim(x.min() - 1, x.max() + 1)
        ax.set_ylim(bottom=1.0)  # speedup ≥ 1 (full res = 1×)
        ax.invert_xaxis()  # decreasing % left to right: 100% on left, 25% on right
        ax.legend(loc="upper left", framealpha=0.95, edgecolor="#cccccc")

        fig.tight_layout(pad=1.0)

        if output_path is None:
            base = os.path.splitext(runtime_and_radius_list_path)[0]
            output_path = os.path.join(os.path.dirname(base), "runtime_figure.pdf")
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
            setup_academic_axes(ax, grid=True)
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


def folder_to_video(
    folder_path,
    output_path=None,
    fps=10,
    image_glob="*.png",
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
        images.append(img)

    with imageio.get_writer(output_path, fps=fps) as writer:
        for img in images:
            writer.append_data(img)

    print(f"Saved: {output_path} ({len(images)} frames @ {fps} fps)")
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
    parser.add_argument("figure", type=str, choices=["runtime", "runtime_quality", "baselines", "foveation_trajectory", "stack", "video"])
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for the figure")
    parser.add_argument("--base-dir", type=str, default=None, help="Base dir (default: ../results/eccv2026_final)")
    parser.add_argument("--padding-frac", type=float, default=0.02, help="Padding between images as fraction of image width")
    parser.add_argument("--downsample", type=int, default=None, help="Downsample factor for foveation_trajectory (optional)")

    # baselines / foveation_trajectory figure-specific arguments
    parser.add_argument("--baselines", type=str, nargs="+", default=["baseline_eval", "naive_eval", "ours_dit_eval"], help="Baseline folder names for baselines figure")
    parser.add_argument("--total-images", type=int, default=20, help="Number of images per baseline")
    parser.add_argument("--num-cols", type=int, default=4, help="Number of images per row")
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="Number of rows for baselines (two prompt triplets per row) or foveation_trajectory (frames per trajectory column)",
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

    # stack figure: list of images, axis, pad_px
    parser.add_argument("--images", type=str, nargs="+", default=None, help="Image paths for stack figure (required when figure=stack)")
    parser.add_argument("--axis", type=str, default="horizontal", choices=["horizontal", "h", "vertical", "v"], help="Stack direction for stack figure: horizontal or vertical")
    parser.add_argument("--pad-px", type=int, default=10, help="Padding in pixels between images for stack figure")

    # video figure: folder of images -> video
    parser.add_argument("--folder", type=str, default=None, help="Folder of images for video figure (required when figure=video)")
    parser.add_argument("--fps", type=float, default=10, help="Frames per second for output video")
    parser.add_argument("--image-glob", type=str, default="*.png", help="Glob for image files in folder (default: *.png)")

    # foveation_trajectory figure-specific arguments
    parser.add_argument("--row1-name", type=str, default="foveation_trajectory_naive", help="Folder name for row 1 of foveation_trajectory (under base-dir)")
    parser.add_argument("--row2-name", type=str, default="foveation_trajectory_ours_dit", help="Folder name for row 2 of foveation_trajectory; use empty string or 'none' to plot only row1")
    parser.add_argument("--subfolder-idx", type=int, nargs="+", default=[0], help="Subfolder index(s), e.g. 0 or 0 1 2 for 000, 001, 002")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_base = os.path.join(script_dir, "..", "results", "eccv2026_final")

    if args.figure == "runtime":
        file_path = os.path.join(default_base, "runtime", "runtime_and_radius_list.npz")
        generate_runtime_figure(file_path, output_path=args.output)
    elif args.figure == "runtime_quality":
        file_path = os.path.join(default_base, "runtime", "runtime_and_radius_list.npz")
        generate_runtime_quality_two_col(file_path, output_path=args.output)
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
        )
    elif args.figure == "stack":
        if not args.images:
            raise ValueError("--images is required for figure=stack (provide one or more image paths)")
        import imageio
        images = [imageio.imread(p) for p in args.images]
        out = args.output or "stacked.png"
        stack_images(
            images,
            output_path=out,
            axis=args.axis,
            pad_px=args.pad_px,
        )
    elif args.figure == "video":
        if not args.folder:
            raise ValueError("--folder is required for figure=video")
        folder_to_video(
            folder_path=args.folder,
            output_path=args.output,
            fps=args.fps,
            image_glob=args.image_glob,
        )
    else:
        raise ValueError(f"Invalid figure: {args.figure}")
