"""Assemble the cfg-4.0 lion-prompt grid into one labelled figure.

Same layout as the earlier SUMMARY_ours_vs_ralu_radii.png: rows = mask radius,
columns = method, per-cell caption with the schedule and the image's mean/std.
Adds the Z=6.5 columns and a full-resolution reference column, and everything here
was generated at cfg_scale 4.0 (see rerun_cfg4.sh).
"""
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/brianchc/foveated_diffusion-private/results/ralu_compare/cfg4/grid"
OUT = "/home/brianchc/foveated_diffusion-private/results/ralu_compare/cfg4/SUMMARY_cfg4_lion.png"

RADII = ["0.1", "0.3", "0.5", "0.7"]
HR_FRAC = {"0.1": "1.6% HR", "0.3": "14.5% HR", "0.5": "38.3% HR", "0.7": "77.0% HR"}
COLS = [
    ("plain",      "FULL-RES  (reference, 50 steps)", "#9ecbff"),
    ("ours",       "OURS  (mixed-res out, 50 steps)", "#a6e22e"),
    ("ralu3_z100", "RALU 3-stage, Z free  (full-res out)", "#e6db74"),
    ("ralu3_z6.5", "RALU 3-stage, Z=6.5  (full-res out)", "#e6db74"),
    ("ralu2_z100", "RALU 2-stage, Z free  (mixed-res out)", "#e6db74"),
    ("ralu2_z6.5", "RALU 2-stage, Z=6.5  (mixed-res out)", "#e6db74"),
]

CELL = 360
PAD = 12
CAP_H = 46
LABEL_W = 130
TITLE_H = 34
HEAD_H = 26
BG = "#181818"

SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"


def font(sz):
    try:
        return ImageFont.truetype(SANS, sz)
    except OSError:
        return ImageFont.load_default()


def cell_dir(tag, radius):
    """plain is radius-independent."""
    return os.path.join(ROOT, "plain" if tag == "plain" else f"{tag}_r{radius}")


def main():
    f_title, f_head, f_cap, f_row = font(17), font(15), font(13), font(16)

    W = LABEL_W + len(COLS) * (CELL + PAD) + PAD
    H = TITLE_H + HEAD_H + len(RADII) * (CELL + CAP_H + PAD) + PAD
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)

    d.text((10, TITLE_H // 2),
           "Lion prompt | circular mask | lr_factor 2 | cfg_scale 4.0 | "
           "RALU compute-matched to ours@50 at each radius",
           fill="#e8e8e8", font=f_title, anchor="lm")

    for c, (_, head, colr) in enumerate(COLS):
        x = LABEL_W + c * (CELL + PAD)
        d.text((x, TITLE_H + HEAD_H // 2), head, fill=colr, font=f_head, anchor="lm")

    for r, rad in enumerate(RADII):
        y = TITLE_H + HEAD_H + r * (CELL + CAP_H + PAD)
        d.text((10, y + CELL // 2 - 10), f"r = {rad}", fill="#e8e8e8", font=f_row, anchor="lm")
        d.text((10, y + CELL // 2 + 12), HR_FRAC[rad], fill="#9a9a9a", font=f_cap, anchor="lm")

        for c, (tag, _, _) in enumerate(COLS):
            x = LABEL_W + c * (CELL + PAD)
            dd = cell_dir(tag, rad)
            hits = sorted(glob.glob(os.path.join(dd, "img_*.png")))
            if not hits:
                d.text((x + 8, y + CELL // 2), "missing", fill="#ff5555", font=f_cap)
                continue
            im = Image.open(hits[0]).convert("RGB")
            arr = np.asarray(im, dtype=np.float32)
            sheet.paste(im.resize((CELL, CELL), Image.LANCZOS), (x, y))

            st = os.path.join(dd, "stats_00000.json")
            if os.path.exists(st):
                s = json.load(open(st))
                N = s["params"]["N"]
                cr = s.get("schedule_info", {}).get("cost_ratio")
                line1 = (f"N={N} ({sum(N)} st)"
                         + (f"  cost x{cr:.4f}" if cr else "")
                         + f"  Z={s['params']['Z']:.2f}")
            else:
                line1 = "50 steps"
            d.text((x, y + CELL + 6), line1, fill="#cfcfcf", font=f_cap)
            d.text((x, y + CELL + 24),
                   f"mean {arr.mean():.1f}  std {arr.std():.1f}",
                   fill="#7fb7ff", font=f_cap)

    sheet.save(OUT)
    print("wrote", OUT, sheet.size)


if __name__ == "__main__":
    main()
