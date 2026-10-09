#!/usr/bin/env python
"""Join the four baselines of a picked set into one labelled side-by-side video.

The horizontal join is *not* reimplemented here: this imports and calls
``_stack_single_set`` from

    /home/brianchc/fov_video/examples/wanvideo/model_training/validate_foveation/postprocess_videos.py

(the ``stack`` subcommand behind ``eval_scripts/stack_videos.sh``), so the
resize-to-common-height and white ``--margin`` logic is exactly the one already
used for the other stacked videos. This script only adds what that script has
no notion of: a fixed method order, a white label band on top, and the prompt
list dropped next to the results.

Column order is fixed to the paper's: **full-res, token merging, naive
mixed-resolution, ours**.

Usage
-----
    conda activate imp_tome
    python scripts/join_baseline_videos.py \
        --set /local/brian/foveated_diffusion_drive/eval200 \
        --picks results/video_picks/eval200/picks.txt \
        --out results/joined_videos/eval200_circle \
        [--variant videos_with_circle] [--margin 10] [--prompts <csv>]

``--prompts-only`` rewrites ``prompts.csv`` / ``ids.txt`` in an existing output
folder without re-encoding anything, for when the prompt csv shows up late.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

STACKER = Path("/home/brianchc/fov_video/examples/wanvideo/model_training"
               "/validate_foveation/postprocess_videos.py")

# (method dir, column label), left to right. This is the paper's order and is
# deliberately not the alphabetical / picker order.
COLUMNS = [
    ("fullres", "Full-Res"),
    ("tome", "Token Merging"),
    ("naive", "Naive Mixed-Resolution"),
    ("ours", "Ours"),
]

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def load_stacker(path: Path):
    """Import the fov_video post-processing module by path."""
    if not path.is_file():
        raise SystemExit(f"stacking script not found: {path}\n"
                         f"pass --stacker <path to postprocess_videos.py>")
    spec = importlib.util.spec_from_file_location("fov_postprocess", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fov_postprocess"] = mod
    spec.loader.exec_module(mod)
    return mod


def find_videos_dir(set_root: Path, method: str, variant: str) -> Path | None:
    """``<set>/<method>/<run>/<step>/<variant>``, globbed over run and step."""
    hits = sorted(p for p in (set_root / method).glob(f"*/*/{variant}") if p.is_dir())
    return hits[-1] if hits else None


def resolve_sources(set_root: Path, variant: str) -> list[tuple[str, str, Path]]:
    """[(method, label, videos dir)] in column order, with the same variant
    fallback the picker uses: fullres has no overlay render, so it falls back
    to its plain videos rather than dropping the column."""
    out = []
    for method, label in COLUMNS:
        d = find_videos_dir(set_root, method, variant)
        if d is None and variant != "videos":
            d = find_videos_dir(set_root, method, "videos")
        if d is None:
            raise SystemExit(f"no videos dir for method {method!r} under {set_root}")
        out.append((method, label, d))
    return out


def column_layout(paths: list[Path], margin: int) -> tuple[int, list[int]]:
    """Height and per-column widths after the stacker's resize, so the labels
    land over the columns they name. Mirrors ``_stack_single_set``: every clip
    is scaled to the *smallest* height among them."""
    sizes = []
    for p in paths:
        with imageio.get_reader(str(p)) as r:
            w, h = r.get_meta_data()["size"]
        sizes.append((int(w), int(h)))
    target_h = min(h for _, h in sizes)
    widths = [w if h == target_h else int(round(w * target_h / float(h)))
              for w, h in sizes]
    return target_h, widths


def label_band(widths: list[int], margin: int, labels: list[str],
               height: int, font_size: int) -> np.ndarray:
    """A white strip with each label centred over its column."""
    total = sum(widths) + margin * (len(widths) - 1)
    img = Image.new("RGB", (total, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = None
    for cand in FONT_CANDIDATES:
        if Path(cand).is_file():
            font = ImageFont.truetype(cand, font_size)
            break
    if font is None:                      # no ttf on the box; PIL's bitmap font
        font = ImageFont.load_default()
    x = 0
    for w, text in zip(widths, labels):
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        draw.text((x + (w - tw) / 2 - box[0], (height - th) / 2 - box[1]),
                  text, fill=(0, 0, 0), font=font)
        x += w + margin
    return np.asarray(img, dtype=np.uint8)


def _pad_to_macroblock(frame: np.ndarray, block: int = 16) -> np.ndarray:
    """Pad right/bottom with white up to a multiple of `block`.

    Without this the h264 writer silently *resizes* an odd-sized frame to fit
    the macro block, which would stretch the labels off their columns.
    """
    h, w = frame.shape[:2]
    ph, pw = (-h) % block, (-w) % block
    if not ph and not pw:
        return frame
    return np.pad(frame, ((0, ph), (0, pw), (0, 0)), constant_values=255)


def add_label_band(joined: Path, out_path: Path, band: np.ndarray,
                   quality: int) -> None:
    """Prepend the label band to every frame of an already-joined video."""
    with imageio.get_reader(str(joined)) as reader:
        fps = reader.get_meta_data().get("fps", 15.0)
        writer = imageio.get_writer(str(out_path), fps=fps, quality=quality,
                                    macro_block_size=1)
        try:
            for frame in reader:
                if frame.shape[1] != band.shape[1]:
                    # The stacker's own writer may have nudged the width to a
                    # macro block; pad the band (in white) to match rather than
                    # let the vstack fail on a 2px disagreement.
                    b = band[:, :frame.shape[1]] if band.shape[1] > frame.shape[1] else \
                        np.pad(band, ((0, 0), (0, frame.shape[1] - band.shape[1]), (0, 0)),
                               constant_values=255)
                else:
                    b = band
                writer.append_data(
                    _pad_to_macroblock(np.concatenate([b, frame[..., :3]], axis=0)))
        finally:
            writer.close()


def read_picks(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_prompts(path: Path) -> dict[str, str]:
    """``[id,]prompt`` -> {id: prompt}; no id column means 1-indexed by row."""
    out: dict[str, str] = {}
    with path.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=1):
            keys = {k.lower(): v for k, v in row.items() if k}
            prompt = keys.get("prompt") or keys.get("text") or ""
            sid = keys.get("id") or keys.get("idx") or keys.get("index") or str(i)
            if prompt:
                out[str(sid).strip()] = prompt
    return out


def write_prompt_list(out_dir: Path, ids: list[str], prompts: dict[str, str],
                      set_root: Path, variant: str) -> None:
    """The picked ids and their prompts, next to the videos they belong to."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ids.txt").write_text("".join(f"{sid}\n" for sid in ids))
    with (out_dir / "prompts.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "id", "video", "prompt"])
        for i, sid in enumerate(ids, 1):
            w.writerow([i, sid, f"{sid}.mp4", prompts.get(sid, "")])
    with (out_dir / "prompts.txt").open("w") as fh:
        fh.write(f"# {set_root}  ({variant})\n")
        fh.write(f"# columns: {' | '.join(label for _, label in COLUMNS)}\n\n")
        for i, sid in enumerate(ids, 1):
            fh.write(f"[{i:2d}] {sid}.mp4\n     {prompts.get(sid, '<prompt unknown>')}\n")
    missing = [s for s in ids if s not in prompts]
    if prompts and missing:
        print(f"  warning: no prompt for ids {' '.join(missing)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_root", type=Path, required=True,
                    help="eval set dir, e.g. /local/brian/foveated_diffusion_drive/eval200")
    ap.add_argument("--picks", type=Path, required=True,
                    help="picks.txt from the video picker (one id per line)")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--variant", default="videos_with_circle",
                    choices=["videos_with_circle", "videos"],
                    help="which render to join (default: the fovea-circle one)")
    ap.add_argument("--margin", type=int, default=10,
                    help="white gap in px between columns (default 10)")
    ap.add_argument("--quality", type=int, default=8,
                    help="imageio/ffmpeg quality 1-10 (default 8)")
    ap.add_argument("--label-height", type=int, default=0,
                    help="label band height in px (default: 11%% of frame height)")
    ap.add_argument("--font-size", type=int, default=0,
                    help="label font size (default: 55%% of the band height)")
    ap.add_argument("--prompts", type=Path, default=None,
                    help="csv with columns [id,]prompt for the prompt list")
    ap.add_argument("--prompts-only", action="store_true",
                    help="only (re)write the prompt list in --out; encode nothing")
    ap.add_argument("--stacker", type=Path, default=STACKER,
                    help="path to fov_video's postprocess_videos.py")
    args = ap.parse_args()

    set_root = args.set_root.expanduser().resolve()
    ids = read_picks(args.picks)
    prompts = read_prompts(args.prompts) if args.prompts else {}
    out_dir = args.out.expanduser().resolve()

    if args.prompts_only:
        write_prompt_list(out_dir, ids, prompts, set_root, args.variant)
        print(f"wrote prompt list for {len(ids)} ids -> {out_dir}")
        return

    stacker = load_stacker(args.stacker)
    sources = resolve_sources(set_root, args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"set     : {set_root}")
    print(f"columns : " + ", ".join(
        f"{label} [{d.parent.parent.name}/{d.parent.name}/{d.name}]"
        for _, label, d in sources))
    print(f"ids     : {' '.join(ids)}")
    print(f"out     : {out_dir}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="join_baselines_"))
    try:
        for i, sid in enumerate(ids, 1):
            paths = [d / f"{sid}.mp4" for _, _, d in sources]
            missing = [p for p in paths if not p.is_file()]
            if missing:
                print(f"[{i}/{len(ids)}] {sid}: missing {', '.join(str(m) for m in missing)}"
                      f" — skipped")
                continue
            joined = tmp_dir / f"{sid}.mp4"
            # The horizontal join itself, straight from the fov_video script.
            stacker._stack_single_set([str(p) for p in paths], str(joined),
                                      margin=args.margin, quality=args.quality)
            height, widths = column_layout(paths, args.margin)
            band_h = args.label_height or max(28, int(round(height * 0.11)))
            band_h += band_h % 2                     # keep the frame even for h264
            font_px = args.font_size or max(12, int(round(band_h * 0.55)))
            band = label_band(widths, args.margin,
                              [label for _, label, _ in sources], band_h, font_px)
            out_path = out_dir / f"{sid}.mp4"
            add_label_band(joined, out_path, band, args.quality)
            size_mb = out_path.stat().st_size / 1e6
            print(f"[{i}/{len(ids)}] {sid}.mp4  "
                  f"{sum(widths) + args.margin * 3}x{height + band_h}  {size_mb:.1f} MB")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    write_prompt_list(out_dir, ids, prompts, set_root, args.variant)
    print(f"done -> {out_dir} ({len(list(out_dir.glob('*.mp4')))} videos, "
          f"prompts.csv / prompts.txt / ids.txt)")


if __name__ == "__main__":
    main()
