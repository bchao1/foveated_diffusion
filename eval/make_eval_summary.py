"""Summary figure for an eval.sh output tree.

Reads $OUT_ROOT/<mask_tag>/<method>/img_%010d.png (+ $OUT_ROOT/high_res, which is
mask-independent) and lays the methods out as columns, prompts as rows, with the foveal
boundary drawn from the same geometry the run used.

    python make_eval_summary.py --out_root ../results/eval_smoke_5arm \
        --mask_tag circular_r0.25_randcenter42 --num_prompts 2

The mask tag encodes the config, so shape / radius / center seed are parsed from it:
    <shape>_r<radius>[_randcenter<seed>]
"""
import argparse
import os
import re
import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import foveation_center_for_index, random_mask_center_range  # noqa: E402

FONT = "/home/brianchc/fonts/cmu-serif/cmunbx.ttf"
LABELS = {"high_res": "high-res reference", "naive": "naive mixed-res", "ours": "ours",
          "tome": "ToMe-2x2", "ralu": "RALU 3-stage"}


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except Exception:
        return ImageFont.load_default()


def parse_tag(tag):
    """<shape>_r<radius>[_randcenter<seed>] -> (shape, radius, seed or None)."""
    m = re.fullmatch(r"(square|circular)_r([0-9.]+)(?:_randcenter(\d+))?", tag)
    if not m:
        raise ValueError(f"cannot parse mask tag {tag!r}")
    return m.group(1), float(m.group(2)), (int(m.group(3)) if m.group(3) else None)


def method_dir(out_root, tag, method):
    # high_res ignores the mask, so eval.sh keeps one shared copy outside the tag dir.
    return os.path.join(out_root, "high_res") if method == "high_res" \
        else os.path.join(out_root, tag, method)


def main(args):
    shape, radius, seed = parse_tag(args.mask_tag)
    lr = args.lr_downsample_factor
    if seed is None:
        centers = [(0.0, 0.0)] * args.num_prompts
        cap = (0.0, 0.0)
    else:
        cap = random_mask_center_range(args.mask_center_range, radius, shape,
                                       args.height, args.width, lr)
        centers = [foveation_center_for_index(i, seed, cap) for i in range(args.num_prompts)]

    methods = args.methods.split()
    prompts_csv = os.path.join(method_dir(args.out_root, args.mask_tag, methods[-1]),
                               "metadata_00000.csv")
    prompts = list(pd.read_csv(prompts_csv)["prompt"]) if os.path.exists(prompts_csv) else []

    tile, pad, hdr, rowlbl = args.tile, 8, 104, 118
    W = rowlbl + len(methods) * (tile + pad) + pad
    H = hdr + args.num_prompts * (tile + pad) + pad
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(sheet)

    title = (f"eval.sh  |  {shape} r={radius}, lr_factor={lr}, {args.steps} steps, cfg {args.cfg}, "
             f"seed {args.seed}  |  " +
             (f"randomized mask center (seed {seed}, range +/-{cap[0]:.3f}, clamped to frame)"
              if seed is not None else "centered mask"))
    d.text((rowlbl, 18), title, fill=(235, 235, 240), font=font(21))

    f_hdr = font(26)
    for c, method in enumerate(methods):
        label = LABELS.get(method, method)
        x = rowlbl + c * (tile + pad)
        d.text((x + (tile - d.textlength(label, font=f_hdr)) / 2, hdr - 38), label,
               fill=(255, 255, 255), font=f_hdr)

    for row in range(args.num_prompts):
        y = hdr + pad + row * (tile + pad)
        cx, cy = centers[row]
        d.text((6, y + 6), f"#{row}", fill=(230, 230, 235), font=font(20))
        if seed is not None:
            d.text((6, y + 30), f"({cx:+.2f},", fill=(150, 150, 158), font=font(16))
            d.text((6, y + 50), f" {cy:+.2f})", fill=(150, 150, 158), font=font(16))
        for c, method in enumerate(methods):
            p = os.path.join(method_dir(args.out_root, args.mask_tag, method),
                             f"img_{row:010d}.png")
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB").resize((tile, tile), Image.LANCZOS)
            if method != "high_res":     # high_res has no fovea to outline
                px, py = (cx + 0.5) * tile, (cy + 0.5) * tile
                dr = ImageDraw.Draw(im)
                if shape == "circular":
                    rad = radius * (2 ** 0.5) / 2 * tile
                    dr.ellipse([px - rad, py - rad, px + rad, py + rad],
                               outline=(255, 60, 60), width=3)
                else:
                    half = radius * tile / 2
                    dr.rectangle([px - half, py - half, px + half, py + half],
                                 outline=(255, 60, 60), width=3)
            sheet.paste(im, (rowlbl + c * (tile + pad), y))

    out_png = args.output or os.path.join(args.out_root, f"summary_{args.mask_tag}.png")
    sheet.save(out_png)
    print(f"wrote {out_png} {sheet.size}")

    if prompts:
        with open(os.path.join(args.out_root, "PROMPTS.md"), "w") as fh:
            for i, p in enumerate(prompts[:args.num_prompts]):
                fh.write(f"{i}. {p}\n\n")
        print(f"wrote {os.path.join(args.out_root, 'PROMPTS.md')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Summary figure for an eval.sh output tree")
    p.add_argument("--out_root", required=True)
    p.add_argument("--mask_tag", required=True,
                   help="<shape>_r<radius>[_randcenter<seed>], the eval.sh subdirectory name")
    p.add_argument("--methods", default="high_res naive ours tome ralu")
    p.add_argument("--num_prompts", type=int, default=6)
    p.add_argument("--mask_center_range", type=float, default=0.3)
    p.add_argument("--lr_downsample_factor", type=int, default=2)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile", type=int, default=340)
    p.add_argument("--output", default=None)
    main(p.parse_args())
