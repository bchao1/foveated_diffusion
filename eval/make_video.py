"""
make_video.py — Video composition utilities.

Functions:
  stack_videos(video_paths, output_path, grid_x, grid_y, margin, bg_color, target_size)
      Arrange videos in an X×Y grid with a margin between them.
      Output is scaled to approximately target_size (default 1920×1080) maintaining aspect ratio.

  concat_videos(video_paths, output_path, sleep_duration, fps, freeze)
      Concatenate videos sequentially with a gap between each (black or frozen last frame).

  add_text_overlay(video_path, output_path, text, font_path, font_size, color, position, padding)
      Burn text onto every frame of a video with configurable font, color, size, and position.

  wipe_video(img_folder, mask_folder, output_path, ...)
      Rose-trajectory video that wipes to a tokenization mask at the midpoint and back.
      Text labels slide with the wipe slider.

Dependencies: imageio, imageio-ffmpeg, numpy, Pillow
"""

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT_DIR = "/home/brianchc/fonts/cmu-serif"
DEFAULT_FONT_PATH = f"{DEFAULT_FONT_DIR}/cmunrm.ttf"
DEFAULT_BOLD_FONT_PATH = f"{DEFAULT_FONT_DIR}/cmunbx.ttf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_video(path) -> tuple[list[np.ndarray], float]:
    """Return (frames, fps). Frames are HxWx3 uint8 RGB arrays."""
    reader = imageio.get_reader(str(path), format="ffmpeg")
    meta = reader.get_meta_data()
    fps = meta.get("fps", 24.0)
    frames = [frame for frame in reader]
    reader.close()
    return frames, fps


def _make_writer(output_path, fps):
    return imageio.get_writer(
        str(output_path),
        format="ffmpeg",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )


def _round_even(x: float) -> int:
    """Round to nearest even integer (required by many codecs)."""
    v = int(round(x))
    return v if v % 2 == 0 else v - 1


def _resize_frames(frames: list[np.ndarray], w: int, h: int) -> list[np.ndarray]:
    """Resize all frames to (w, h) using Lanczos. Returns new list."""
    return [
        np.array(Image.fromarray(f).resize((w, h), Image.LANCZOS))
        for f in frames
    ]


# ---------------------------------------------------------------------------
# 1. Stack videos in an X×Y grid
# ---------------------------------------------------------------------------

def stack_videos(
    video_paths: list[str | Path],
    output_path: str | Path,
    grid_x: int,
    grid_y: int,
    margin: int = 10,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    target_size: tuple[int, int] | None = (1920, 1080),
) -> None:
    """
    Arrange videos in a grid_x × grid_y layout and write to output_path.

    The output canvas is scaled so it fits within target_size (width × height)
    while maintaining aspect ratio. Pass target_size=None to skip scaling.

    Args:
        video_paths:  List of paths to input videos. Must have <= grid_x * grid_y entries.
        output_path:  Path for the output .mp4 file.
        grid_x:       Number of columns.
        grid_y:       Number of rows.
        margin:       Pixel gap between cells (no border around edges).
        bg_color:     RGB background fill color (default: black).
        target_size:  (target_w, target_h) to scale the output to. Default: (1920, 1080).
    """
    n = len(video_paths)
    if n > grid_x * grid_y:
        raise ValueError(
            f"{n} videos but grid is only {grid_x}×{grid_y} = {grid_x * grid_y} cells"
        )

    all_frames = []
    fps_list = []
    for p in video_paths:
        frames, fps = _read_video(p)
        all_frames.append(frames)
        fps_list.append(fps)

    out_fps = fps_list[0]
    native_h, native_w = all_frames[0][0].shape[:2]

    # Compute native canvas size
    native_canvas_w = grid_x * native_w + (grid_x - 1) * margin
    native_canvas_h = grid_y * native_h + (grid_y - 1) * margin

    # Scale cell dimensions so the canvas fits within target_size
    if target_size is not None:
        target_w, target_h = target_size
        scale = min(target_w / native_canvas_w, target_h / native_canvas_h)
    else:
        scale = 1.0

    cell_w = _round_even(native_w * scale)
    cell_h = _round_even(native_h * scale)
    scaled_margin = int(round(margin * scale))

    canvas_w = _round_even(grid_x * cell_w + (grid_x - 1) * scaled_margin)
    canvas_h = _round_even(grid_y * cell_h + (grid_y - 1) * scaled_margin)

    # Resize all frames upfront to avoid per-frame PIL overhead in the write loop
    print(f"Resizing {n} clips to cell size {cell_w}×{cell_h} (scale={scale:.3f})...")
    all_frames = [_resize_frames(frames, cell_w, cell_h) for frames in all_frames]

    total_frames = max(len(f) for f in all_frames)
    writer = _make_writer(output_path, out_fps)

    try:
        for t in range(total_frames):
            canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
            for idx, frames in enumerate(all_frames):
                frame = frames[min(t, len(frames) - 1)]
                col = idx % grid_x
                row = idx // grid_x
                x0 = col * (cell_w + scaled_margin)
                y0 = row * (cell_h + scaled_margin)
                x1 = min(x0 + cell_w, canvas_w)
                y1 = min(y0 + cell_h, canvas_h)
                canvas[y0:y1, x0:x1] = frame[:y1 - y0, :x1 - x0]
            writer.append_data(canvas)
    finally:
        writer.close()

    print(f"Stacked video saved to {output_path}  ({canvas_w}×{canvas_h} @ {out_fps:.1f} fps)")


# ---------------------------------------------------------------------------
# 2. Concatenate videos with a black sleep gap between each
# ---------------------------------------------------------------------------

def concat_videos(
    video_paths: list[str | Path],
    output_path: str | Path,
    sleep_duration: float = 1.0,
    fps: float | None = None,
    freeze: bool = False,
) -> None:
    """
    Concatenate videos one after another, inserting a pause between each.

    All videos are resized to the dimensions of the first video.

    Args:
        video_paths:     List of paths to input videos (in order).
        output_path:     Path for the output .mp4 file.
        sleep_duration:  Duration of the gap between clips, in seconds.
        fps:             Output frame rate. Defaults to the first video's fps.
        freeze:          If True, hold the last frame of each clip during the gap.
                         If False (default), show a black frame.
    """
    all_frames = []
    fps_list = []
    for p in video_paths:
        frames, f = _read_video(p)
        all_frames.append(frames)
        fps_list.append(f)

    out_fps = fps if fps is not None else fps_list[0]
    cell_h, cell_w = all_frames[0][0].shape[:2]
    sleep_frames = max(1, int(math.ceil(sleep_duration * out_fps)))

    # Normalize all clips to the reference cell size upfront
    all_frames = [
        _resize_frames(frames, cell_w, cell_h) if frames[0].shape[:2] != (cell_h, cell_w) else frames
        for frames in all_frames
    ]

    writer = _make_writer(output_path, out_fps)

    try:
        for clip_idx, frames in enumerate(all_frames):
            for frame in frames:
                writer.append_data(frame)

            # Insert gap after every clip except the last
            if clip_idx < len(all_frames) - 1:
                gap_frame = frames[-1] if freeze else np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                for _ in range(sleep_frames):
                    writer.append_data(gap_frame)
    finally:
        writer.close()

    gap_type = "freeze" if freeze else "black"
    print(
        f"Concatenated video saved to {output_path}  "
        f"({cell_w}×{cell_h} @ {out_fps:.1f} fps, "
        f"{sleep_frames}-frame {gap_type} gap between clips)"
    )


# ---------------------------------------------------------------------------
# 3. Add text overlay to a video
# ---------------------------------------------------------------------------

_POSITIONS = {"top-left", "top-center", "top-right", "center", "bottom-left", "bottom-center", "bottom-right"}


def _draw_text_on_frame(
    frame: np.ndarray,
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
    position: str,
    padding: int,
    outline_width: int = 0,
) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    W, H = img.size
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=outline_width)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if position == "top-left":
        xy = (padding, padding)
    elif position == "top-center":
        xy = ((W - tw) // 2, padding)
    elif position == "top-right":
        xy = (W - tw - padding, padding)
    elif position == "center":
        xy = ((W - tw) // 2, (H - th) // 2)
    elif position == "bottom-left":
        xy = (padding, H - th - padding)
    elif position == "bottom-center":
        xy = ((W - tw) // 2, H - th - padding)
    elif position == "bottom-right":
        xy = (W - tw - padding, H - th - padding)
    else:
        raise ValueError(f"Unknown position '{position}'. Choose from: {_POSITIONS}")

    draw.text(xy, text, font=font, fill=color,
              stroke_width=outline_width, stroke_fill=(0, 0, 0) if outline_width > 0 else None)
    return np.array(img)


def _make_header_bar(text: str, width: int, font: ImageFont.FreeTypeFont,
                     color: tuple, padding: int, outline_width: int,
                     bar_h: int | None = None) -> np.ndarray:
    """Create a black horizontal bar with `text` pixel-centered in it.

    If bar_h is None, the height is inferred from text metrics + padding.
    Pass bar_h explicitly to enforce a shared fixed height across multiple bars.
    """
    dummy = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font, stroke_width=outline_width)
    text_h = bbox[3] - bbox[1]
    text_w = bbox[2] - bbox[0]
    if bar_h is None:
        bar_h = _round_even(text_h + 2 * padding)
    bar = Image.new("RGB", (width, bar_h), (0, 0, 0))
    draw = ImageDraw.Draw(bar)
    # Subtract bbox offsets so rendered pixels (not the anchor) are truly centered
    xy = ((width - text_w) // 2 - bbox[0], (bar_h - text_h) // 2 - bbox[1])
    draw.text(xy, text, font=font, fill=color,
              stroke_width=outline_width, stroke_fill=(0, 0, 0) if outline_width > 0 else None)
    return np.array(bar)


def add_text_overlay(
    video_path: str | Path,
    output_path: str | Path,
    text: str,
    font_path: str | Path = DEFAULT_FONT_PATH,
    font_size: int = 48,
    color: tuple[int, int, int] = (255, 255, 255),
    position: str = "bottom-center",
    padding: int = 20,
    outline_width: int = 0,
    header: bool = False,
    header_height: int | None = None,
) -> None:
    """
    Burn text onto every frame of a video, or prepend a black header bar above each frame.

    Args:
        video_path:     Input video path.
        output_path:    Output video path.
        text:           Text string to render on each frame.
        font_path:      Path to a .ttf font file. Default: CMU Serif Regular.
        font_size:      Font size in points. Default: 48.
        color:          RGB text color. Default: white (255, 255, 255).
        position:       One of: top-left, top-center, top-right,
                        center, bottom-left, bottom-center, bottom-right.
                        Ignored when header=True.
        padding:        Pixel margin from the frame edge (overlay) or inside bar (header). Default: 20.
        outline_width:  Black outline stroke width in pixels. 0 = no outline. Default: 0.
        header:         If True, prepend a black label bar above the video instead of
                        overlaying text on the frame. Output video is taller by bar height.
        header_height:  Fix the bar height in pixels (useful to match bars across multiple videos).
                        If None (default), height is inferred from text metrics + padding.
    """
    frames, fps = _read_video(video_path)
    font = ImageFont.truetype(str(font_path), font_size)
    h, w = frames[0].shape[:2]

    if header:
        bar = _make_header_bar(text, w, font, color, padding, outline_width, bar_h=header_height)
        bar_h = bar.shape[0]
        writer = _make_writer(output_path, fps)
        try:
            for frame in frames:
                writer.append_data(np.concatenate([bar, frame], axis=0))
        finally:
            writer.close()
        print(f"Header bar saved to {output_path}  ({w}×{h + bar_h} @ {fps:.1f} fps, text='{text}')")
    else:
        writer = _make_writer(output_path, fps)
        try:
            for frame in frames:
                writer.append_data(_draw_text_on_frame(frame, text, font, color, position, padding, outline_width))
        finally:
            writer.close()
        print(f"Text overlay saved to {output_path}  ({w}×{h} @ {fps:.1f} fps, text='{text}', pos={position})")


# ---------------------------------------------------------------------------
# 4. Images → video
# ---------------------------------------------------------------------------

def _resolve_grid_dims(n: int, grid_rows: int | None, grid_cols: int | None) -> tuple[int, int]:
    """Infer grid dimensions from image count when not both provided."""
    if grid_rows is None and grid_cols is None:
        side = int(round(math.sqrt(n)))
        return side, side
    elif grid_rows is None:
        return math.ceil(n / grid_cols), grid_cols
    elif grid_cols is None:
        return grid_rows, math.ceil(n / grid_rows)
    return grid_rows, grid_cols


def _curve_to_grid_order(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    all_paths: list[Path],
    grid_rows: int,
    grid_cols: int,
) -> list[Path]:
    """Normalize a centered curve (x_raw, y_raw) to grid coordinates and return image paths."""
    cx = (grid_cols - 1) / 2.0
    cy = (grid_rows - 1) / 2.0
    x_max = float(np.max(np.abs(x_raw))) or 1.0
    y_max = float(np.max(np.abs(y_raw))) or 1.0
    scale = min(cx / x_max, cy / y_max) * 0.95
    cols = np.clip(np.round(cx + x_raw * scale).astype(int), 0, grid_cols - 1)
    rows = np.clip(np.round(cy + y_raw * scale).astype(int), 0, grid_rows - 1)
    indices = rows * grid_cols + cols
    n = len(all_paths)
    return [all_paths[int(idx) % n] for idx in indices]


# ---------------------------------------------------------------------------
# Trajectory generators — each returns (x_raw, y_raw) centered at origin
# ---------------------------------------------------------------------------

def _traj_lissajous(num_frames, a=3.0, b=2.0, delta=math.pi / 4, **_):
    """Lissajous figure: x=sin(at+δ), y=sin(bt)."""
    t = np.linspace(0, 2 * math.pi, num_frames, endpoint=False)
    return np.sin(a * t + delta), np.sin(b * t)


def _traj_spiral(num_frames, **_):
    """Archimedean spiral: expands outward then contracts back (2 rotations each way)."""
    t = np.linspace(0, 1, num_frames, endpoint=False)
    r = np.sin(math.pi * t)                   # bell: 0 → 1 → 0
    theta = t * 4 * math.pi                   # 2 full rotations
    return r * np.cos(theta), r * np.sin(theta)


def _traj_rose(num_frames, k=4, **_):
    """Rhodonea (rose) curve: r = cos(kθ).  k=4 gives 8 petals."""
    theta = np.linspace(0, 2 * math.pi, num_frames, endpoint=False)
    r = np.cos(k * theta)
    return r * np.cos(theta), r * np.sin(theta)


def _traj_epitrochoid(num_frames, R=3, r=1, d=2, **_):
    """Epitrochoid (outer spirograph): R=3,r=1,d=2 → 3-lobed rounded star."""
    theta = np.linspace(0, 2 * math.pi * r, num_frames, endpoint=False)
    x = (R + r) * np.cos(theta) - d * np.cos((R + r) / r * theta)
    y = (R + r) * np.sin(theta) - d * np.sin((R + r) / r * theta)
    return x, y


def _traj_hypotrochoid(num_frames, R=5, r=3, d=5, **_):
    """Hypotrochoid (inner spirograph): R=5,r=3,d=5 → 5-pointed star rosette."""
    # Full period: 2π × lcm(R,r)/R = 6π for R=5,r=3
    theta = np.linspace(0, 2 * math.pi * r, num_frames, endpoint=False)
    x = (R - r) * np.cos(theta) + d * np.cos((R - r) / r * theta)
    y = (R - r) * np.sin(theta) - d * np.sin((R - r) / r * theta)
    return x, y


def _traj_butterfly(num_frames, **_):
    """Temple H. Fay butterfly curve: r = e^cos(θ) − 2cos(4θ) + sin⁵((2θ−π)/24)."""
    theta = np.linspace(0, 12 * math.pi, num_frames, endpoint=False)
    r = (np.exp(np.cos(theta))
         - 2 * np.cos(4 * theta)
         + np.sin((2 * theta - math.pi) / 24) ** 5)
    return r * np.cos(theta), r * np.sin(theta)


_TRAJECTORY_FNS = {
    "lissajous":   _traj_lissajous,
    "spiral":      _traj_spiral,
    "rose":        _traj_rose,
    "epitrochoid": _traj_epitrochoid,
    "hypotrochoid": _traj_hypotrochoid,
    "butterfly":   _traj_butterfly,
}


def _trajectory_order(
    trajectory: str,
    all_paths: list[Path],
    num_frames: int,
    grid_rows: int | None,
    grid_cols: int | None,
    **kwargs,
) -> list[Path]:
    """Dispatch to the requested trajectory generator and map to grid image paths."""
    if trajectory not in _TRAJECTORY_FNS:
        raise ValueError(f"Unknown trajectory '{trajectory}'. Choose from: {list(_TRAJECTORY_FNS)}")
    grid_rows, grid_cols = _resolve_grid_dims(len(all_paths), grid_rows, grid_cols)
    x_raw, y_raw = _TRAJECTORY_FNS[trajectory](num_frames=num_frames, **kwargs)
    return _curve_to_grid_order(x_raw, y_raw, all_paths, grid_rows, grid_cols)


def images_to_video(
    folder: str | Path,
    output_path: str | Path,
    fps: float = 24.0,
    image_order: list[str] | None = None,
    pattern: str = "*.png",
    trajectory: str | None = None,
    num_frames: int | None = None,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
    lissajous_a: float = 3.0,
    lissajous_b: float = 2.0,
    lissajous_delta: float = math.pi / 4,
) -> None:
    """
    Load images from a folder and encode them as a video.

    By default all images matching `pattern` are sorted by filename and used in that order.
    Pass `image_order` to use a specific subset and/or ordering, or set `trajectory="lissajous"`
    to generate a smooth Lissajous curve path through the image grid.

    Args:
        folder:          Directory containing the images.
        output_path:     Output .mp4 path.
        fps:             Output frame rate. Default: 24.
        image_order:     Optional list of filenames or bare IDs determining frame order.
        pattern:         Glob pattern to discover images. Default: "*.png".
        trajectory:      Trajectory type. Currently supports "lissajous". Default: None.
        num_frames:      Number of output frames when using a trajectory. Required if trajectory is set.
        grid_rows:       Grid rows for trajectory. Inferred from sqrt(n) if None.
        grid_cols:       Grid columns for trajectory. Inferred from sqrt(n) if None.
        lissajous_a:     Lissajous frequency ratio for x-axis. Default: 3.
        lissajous_b:     Lissajous frequency ratio for y-axis. Default: 2.
        lissajous_delta: Lissajous phase offset (radians). Default: π/4.
    """
    folder = Path(folder)

    # Discover all candidate images first
    all_paths = sorted(folder.glob(pattern))
    if not all_paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {folder}")

    if trajectory is not None:
        if num_frames is None:
            raise ValueError("--num-frames is required when using --trajectory")
        paths = _trajectory_order(
            trajectory, all_paths, num_frames, grid_rows, grid_cols,
            a=lissajous_a, b=lissajous_b, delta=lissajous_delta,
        )
    elif image_order is not None:
        paths = []
        for name in image_order:
            p = folder / name
            if p.is_file():
                paths.append(p)
            else:
                # Try matching by stem suffix (handles zero-padded names like img_0003.png)
                hit = next((m for m in all_paths if m.stem == name or m.stem.lstrip("0") == name.lstrip("0") or m.name == name), None)
                if hit is None:
                    raise FileNotFoundError(f"No image found for id '{name}' in {folder}")
                paths.append(hit)
    else:
        paths = all_paths

    first = np.array(Image.open(paths[0]).convert("RGB"))
    h, w = first.shape[:2]
    writer = _make_writer(output_path, fps)
    try:
        writer.append_data(first)
        for p in paths[1:]:
            frame = np.array(Image.open(p).convert("RGB"))
            if frame.shape[:2] != (h, w):
                frame = np.array(Image.fromarray(frame).resize((w, h), Image.LANCZOS))
            writer.append_data(frame)
    finally:
        writer.close()

    print(f"Video saved to {output_path}  ({w}×{h} @ {fps:.1f} fps, {len(paths)} frames)")


# ---------------------------------------------------------------------------
# 5. Wipe video — trajectory animation with horizontal slider to mask
# ---------------------------------------------------------------------------

def _label_frame_top_center(
    frame: np.ndarray,
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple,
    padding: int,
    outline_width: int,
) -> np.ndarray:
    """Draw text top-center with correct bbox-offset centering and black outline."""
    img = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(img)
    W_f, _ = img.size
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=outline_width)
    tw = bbox[2] - bbox[0]
    xy = ((W_f - tw) // 2 - bbox[0], padding - bbox[1])
    draw.text(xy, text, font=font, fill=color,
              stroke_width=outline_width, stroke_fill=(0, 0, 0) if outline_width > 0 else None)
    return np.array(img)


def wipe_video(
    img_folder: str | Path,
    mask_folder: str | Path,
    output_path: str | Path,
    fps: float = 30.0,
    num_frames: int = 240,
    grid_rows: int = 10,
    grid_cols: int = 10,
    trajectory: str = "rose",
    wipe_frames: int = 30,
    hold_frames: int = 60,
    freeze_frames: int = 15,
    slider_width: int = 10,
    img_label: str = "Generated Image",
    mask_label: str = "Tokenization",
    font_path: str | Path = DEFAULT_BOLD_FONT_PATH,
    font_size: int = 82,
    color: tuple[int, int, int] = (255, 255, 255),
    outline_width: int = 3,
    padding: int = 20,
    img_pattern: str = "img_*.png",
    mask_pattern: str = "tokenization_mask_*.png",
    mask_offset: int = 0,
) -> None:
    """
    Generate a trajectory animation over img_folder that wipes to the corresponding
    tokenization mask at the midpoint, then wipes back and continues.

    Both text labels are pre-baked onto their respective frames before compositing,
    so they slide with the wipe slider.

    Video structure (default params → 390 frames = 13s @ 30fps):
      0 .. half-1              : trajectory animation — img_label
      half .. half+freeze-1    : freeze image (0.5s) — img_label
      ..+wipe-1                : wipe L→R (1s) — labels composite with slider
      ..+hold-1                : hold mask (2s) — mask_label
      ..+freeze-1              : freeze mask (0.5s) — mask_label
      ..+wipe-1                : wipe R→L (1s) — labels composite with slider
      ..+half-1                : continue trajectory — img_label

    Args:
        img_folder:     Folder of generated images (e.g. 10×10 grid).
        mask_folder:    Folder of tokenization mask images.
        output_path:    Output .mp4 path.
        fps:            Output frame rate. Default: 30.
        num_frames:     Total trajectory frames. Default: 240.
        grid_rows:      Grid rows for trajectory. Default: 10.
        grid_cols:      Grid columns for trajectory. Default: 10.
        trajectory:     Trajectory type (see _TRAJECTORY_FNS). Default: "rose".
        wipe_frames:    Duration of each wipe in frames. Default: 30 (1s @ 30fps).
        hold_frames:    Duration of mask hold in frames. Default: 60 (2s @ 30fps).
        freeze_frames:  Duration of freeze before each wipe. Default: 15 (0.5s @ 30fps).
        slider_width:   Width of the white slider bar in pixels. Default: 10.
        img_label:      Text label for the image side. Default: "Generated Image".
        mask_label:     Text label for the mask side. Default: "Tokenization".
        font_path:      Path to .ttf font. Default: CMU Serif Bold.
        font_size:      Font size in points. Default: 82.
        color:          Text color (RGB). Default: white.
        outline_width:  Black outline stroke width. Default: 3.
        padding:        Pixel padding from the top edge. Default: 20.
        img_pattern:    Glob pattern for image files. Default: "img_*.png".
        mask_pattern:   Glob pattern for mask files. Default: "tokenization_mask_*.png".
        mask_offset:    Integer shift applied to mask indices to correct alignment.
                        E.g. -1 uses the mask one position earlier in sorted order. Default: 0.
    """
    img_folder  = Path(img_folder)
    mask_folder = Path(mask_folder)
    output_path = Path(output_path)

    imgs  = sorted(img_folder.glob(img_pattern))
    masks = sorted(mask_folder.glob(mask_pattern))
    if not imgs:
        raise FileNotFoundError(f"No images matching '{img_pattern}' in {img_folder}")
    if not masks:
        raise FileNotFoundError(f"No masks matching '{mask_pattern}' in {mask_folder}")
    print(f"Found {len(imgs)} images, {len(masks)} masks.")

    # Build trajectory index sequence
    indices = _trajectory_order(
        trajectory, imgs, num_frames, grid_rows, grid_cols,
    )
    # indices is a list of Paths; convert to integer indices into imgs/masks
    idx_list = [imgs.index(p) for p in indices]

    def load_rgb(path):
        return np.array(Image.open(path).convert("RGB"))

    print("Loading frames...")
    img_frames  = [load_rgb(imgs[i % len(imgs)])                        for i in idx_list]
    mask_frames = [load_rgb(masks[(i + mask_offset) % len(masks)]) for i in idx_list]
    print("Done loading.")

    H, W = img_frames[0].shape[:2]
    WHITE = np.full((H, slider_width, 3), 255, dtype=np.uint8)

    font = ImageFont.truetype(str(font_path), font_size)

    def label(frame, text):
        return _label_frame_top_center(frame, text, font, color, padding, outline_width)

    # Pre-label all image frames; mask frames only needed at the midpoint
    img_labeled  = [label(f, img_label) for f in img_frames]
    half = num_frames // 2

    freeze_img_labeled  = img_labeled[half - 1]
    freeze_mask_raw     = mask_frames[half - 1]
    freeze_mask_labeled = label(freeze_mask_raw, mask_label)
    resume_img_labeled  = img_labeled[half]

    def composite(left_labeled, right_labeled, x_div):
        """Composite two labeled frames with a white bar at x_div."""
        out = right_labeled.copy()
        out[:, :x_div] = left_labeled[:, :x_div]
        x0, x1 = x_div, min(x_div + slider_width, W)
        if x0 < x1:
            out[:, x0:x1] = 255
        return out

    def wipe_in(t):
        x_div = int(W * t / (wipe_frames - 1)) if wipe_frames > 1 else W
        return composite(freeze_mask_labeled, freeze_img_labeled, x_div)

    def wipe_out(t):
        x_div = int(W * (wipe_frames - 1 - t) / (wipe_frames - 1)) if wipe_frames > 1 else 0
        return composite(freeze_mask_labeled, resume_img_labeled, x_div)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _make_writer(output_path, fps)
    n = 0
    try:
        for i in range(half):
            writer.append_data(img_labeled[i]); n += 1
        for _ in range(freeze_frames):
            writer.append_data(freeze_img_labeled); n += 1
        for t in range(wipe_frames):
            writer.append_data(wipe_in(t)); n += 1
        for _ in range(hold_frames):
            writer.append_data(freeze_mask_labeled); n += 1
        for _ in range(freeze_frames):
            writer.append_data(freeze_mask_labeled); n += 1
        for t in range(wipe_frames):
            writer.append_data(wipe_out(t)); n += 1
        for i in range(half, num_frames):
            writer.append_data(img_labeled[i]); n += 1
    finally:
        writer.close()

    print(f"Wipe video saved to {output_path}  ({W}×{H} @ {fps:.1f} fps, {n} frames = {n/fps:.1f}s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Video composition utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- stack ---
    s = sub.add_parser("stack", help="Arrange videos in an X×Y grid")
    s.add_argument("videos", nargs="+", help="Input video paths")
    s.add_argument("--output", "-o", required=True, help="Output path")
    s.add_argument("--grid-x", type=int, required=True, help="Number of columns")
    s.add_argument("--grid-y", type=int, required=True, help="Number of rows")
    s.add_argument("--margin", type=int, default=10, help="Pixel gap between videos (default: 10)")
    s.add_argument(
        "--bg-color", type=str, default="0,0,0",
        help="Background RGB color as R,G,B (default: 0,0,0 = black)",
    )
    s.add_argument(
        "--target-size", type=str, default="1920x1080",
        help="Scale output to fit within WxH while maintaining aspect ratio (default: 1920x1080). Pass 'none' to disable.",
    )

    # --- concat ---
    c = sub.add_parser("concat", help="Concatenate videos with a black gap between each")
    c.add_argument("videos", nargs="+", help="Input video paths")
    c.add_argument("--output", "-o", required=True, help="Output path")
    c.add_argument("--sleep", type=float, default=1.0, help="Gap duration in seconds (default: 1.0)")
    c.add_argument("--fps", type=float, default=None, help="Output FPS (default: from first video)")
    c.add_argument("--freeze", action="store_true", default=False, help="Hold last frame of each clip during gap instead of black")

    # --- text ---
    t = sub.add_parser("text", help="Burn text onto every frame of a video")
    t.add_argument("video", help="Input video path")
    t.add_argument("--output", "-o", required=True, help="Output path")
    t.add_argument("--text", required=True, help="Text to overlay")
    t.add_argument("--font", type=str, default=DEFAULT_FONT_PATH, help=f"Path to .ttf font file (default: CMU Serif Regular)")
    t.add_argument("--font-size", type=int, default=48, help="Font size in points (default: 48)")
    t.add_argument("--color", type=str, default="255,255,255", help="Text color as R,G,B (default: 255,255,255 = white)")
    t.add_argument("--position", type=str, default="bottom-center",
                   choices=sorted(_POSITIONS),
                   help="Text anchor position (default: bottom-center)")
    t.add_argument("--padding", type=int, default=20, help="Pixel margin from frame edge (default: 20)")
    t.add_argument("--outline-width", type=int, default=0, help="Black outline stroke width in pixels (default: 0 = none)")
    t.add_argument("--header", action="store_true", default=False,
                   help="Prepend a black label bar above the video instead of overlaying text on the frame")
    t.add_argument("--header-height", type=int, default=None,
                   help="Fix header bar height in pixels (use same value across videos for matched bars)")

    # --- wipe ---
    w = sub.add_parser("wipe", help="Trajectory animation that wipes to a tokenization mask and back")
    w.add_argument("img_folder",  help="Folder of generated images (e.g. 10×10 grid)")
    w.add_argument("mask_folder", help="Folder of tokenization mask images")
    w.add_argument("--output", "-o", required=True, help="Output path")
    w.add_argument("--fps", type=float, default=30.0, help="Output frame rate (default: 30)")
    w.add_argument("--num-frames", type=int, default=240, help="Total trajectory frames (default: 240)")
    w.add_argument("--grid-rows", type=int, default=10, help="Grid rows for trajectory (default: 10)")
    w.add_argument("--grid-cols", type=int, default=10, help="Grid columns for trajectory (default: 10)")
    w.add_argument("--trajectory", type=str, default="rose", choices=list(_TRAJECTORY_FNS),
                   help="Trajectory type (default: rose)")
    w.add_argument("--wipe-frames",   type=int, default=30, help="Frames per wipe (default: 30 = 1s @ 30fps)")
    w.add_argument("--hold-frames",   type=int, default=60, help="Frames to hold mask (default: 60 = 2s @ 30fps)")
    w.add_argument("--freeze-frames", type=int, default=15, help="Frames to freeze before each wipe (default: 15 = 0.5s @ 30fps)")
    w.add_argument("--slider-width",  type=int, default=10, help="White slider bar width in pixels (default: 10)")
    w.add_argument("--img-label",  type=str, default="Generated Image", help="Label for the image side (default: 'Generated Image')")
    w.add_argument("--mask-label", type=str, default="Tokenization",    help="Label for the mask side (default: 'Tokenization')")
    w.add_argument("--font", type=str, default=DEFAULT_BOLD_FONT_PATH, help="Path to .ttf font (default: CMU Serif Bold)")
    w.add_argument("--font-size", type=int, default=82, help="Font size in points (default: 82)")
    w.add_argument("--color", type=str, default="255,255,255", help="Text color as R,G,B (default: white)")
    w.add_argument("--outline-width", type=int, default=3, help="Black outline stroke width (default: 3)")
    w.add_argument("--padding", type=int, default=20, help="Pixel padding from top edge (default: 20)")
    w.add_argument("--img-pattern",  type=str, default="img_*.png",
                   help="Glob pattern for image files (default: img_*.png)")
    w.add_argument("--mask-pattern", type=str, default="tokenization_mask_*.png",
                   help="Glob pattern for mask files (default: tokenization_mask_*.png)")
    w.add_argument("--mask-offset", type=int, default=0,
                   help="Shift mask index by N to correct alignment (e.g. -1 uses previous mask). Default: 0")

    # --- images ---
    i = sub.add_parser("images", help="Compile images from a folder into a video")
    i.add_argument("folder", help="Folder containing images")
    i.add_argument("--output", "-o", required=True, help="Output path")
    i.add_argument("--fps", type=float, default=24.0, help="Output frame rate (default: 24)")
    i.add_argument("--pattern", type=str, default="*.png", help="Glob pattern to match images (default: *.png)")
    i.add_argument("--order", nargs="+", default=None,
                   help="Explicit list of filenames or bare IDs to use, in order. "
                        "E.g.: --order img_003.png img_001.png  or  --order 003 001")
    i.add_argument("--trajectory", type=str, default=None, choices=list(_TRAJECTORY_FNS),
                   help=f"Smooth trajectory through image grid. Choices: {list(_TRAJECTORY_FNS)}")
    i.add_argument("--num-frames", type=int, default=None,
                   help="Number of output frames (required when --trajectory is set)")
    i.add_argument("--grid-rows", type=int, default=None,
                   help="Grid rows for trajectory (default: inferred from sqrt of image count)")
    i.add_argument("--grid-cols", type=int, default=None,
                   help="Grid columns for trajectory (default: inferred from sqrt of image count)")
    i.add_argument("--lissajous-a", type=float, default=3.0,
                   help="Lissajous x-frequency ratio (default: 3)")
    i.add_argument("--lissajous-b", type=float, default=2.0,
                   help="Lissajous y-frequency ratio (default: 2)")
    i.add_argument("--lissajous-delta", type=float, default=math.pi / 4,
                   help="Lissajous phase offset in radians (default: π/4 ≈ 0.785)")

    return parser.parse_args()


def main():
    args = _parse_args()

    if args.command == "stack":
        bg = tuple(int(x) for x in args.bg_color.split(","))
        if args.target_size.lower() == "none":
            target_size = None
        else:
            tw, th = args.target_size.lower().split("x")
            target_size = (int(tw), int(th))
        stack_videos(
            video_paths=args.videos,
            output_path=args.output,
            grid_x=args.grid_x,
            grid_y=args.grid_y,
            margin=args.margin,
            bg_color=bg,
            target_size=target_size,
        )
    elif args.command == "concat":
        concat_videos(
            video_paths=args.videos,
            output_path=args.output,
            sleep_duration=args.sleep,
            fps=args.fps,
            freeze=args.freeze,
        )
    elif args.command == "text":
        color = tuple(int(x) for x in args.color.split(","))
        add_text_overlay(
            video_path=args.video,
            output_path=args.output,
            text=args.text,
            font_path=args.font,
            font_size=args.font_size,
            color=color,
            position=args.position,
            padding=args.padding,
            outline_width=args.outline_width,
            header=args.header,
            header_height=args.header_height,
        )


    elif args.command == "wipe":
        color = tuple(int(x) for x in args.color.split(","))
        wipe_video(
            img_folder=args.img_folder,
            mask_folder=args.mask_folder,
            output_path=args.output,
            fps=args.fps,
            num_frames=args.num_frames,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            trajectory=args.trajectory,
            wipe_frames=args.wipe_frames,
            hold_frames=args.hold_frames,
            freeze_frames=args.freeze_frames,
            slider_width=args.slider_width,
            img_label=args.img_label,
            mask_label=args.mask_label,
            font_path=args.font,
            font_size=args.font_size,
            color=color,
            outline_width=args.outline_width,
            padding=args.padding,
            img_pattern=args.img_pattern,
            mask_pattern=args.mask_pattern,
            mask_offset=args.mask_offset,
        )

    elif args.command == "images":
        images_to_video(
            folder=args.folder,
            output_path=args.output,
            fps=args.fps,
            image_order=args.order,
            pattern=args.pattern,
            trajectory=args.trajectory,
            num_frames=args.num_frames,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            lissajous_a=args.lissajous_a,
            lissajous_b=args.lissajous_b,
            lissajous_delta=args.lissajous_delta,
        )


if __name__ == "__main__":
    main()
