"""
Re-render iou_overlay_0000.png in every subfolder of the saliency_IoU_experiment
directory with circle outlines drawn 2x thicker (line_width 4 -> 8).

Reuses the existing img_0000.png, mask_0000.png (GT), saliency_mask_0000.png
(predicted), and the per-image IoU stored in results.json. Does not re-run
the saliency detector.
"""

import os
import json
import glob
import argparse

import numpy as np
from PIL import Image, ImageDraw

ROOT_DIR = "/home/brianchc/foveated_diffusion/results/eccv2026_rebuttal/saliency_IoU_experiment"
LINE_WIDTH = 8  # 2x the original 4


def circle_from_binary_mask(binary_mask):
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cx = (cmin + cmax) / 2.0
    cy = (rmin + rmax) / 2.0
    radius = ((rmax - rmin) + (cmax - cmin)) / 4.0
    return cx, cy, radius


def draw_iou_overlay(img_np, pred_binary, gt_binary, iou, line_width=LINE_WIDTH):
    img_pil = Image.fromarray(img_np).convert("RGB")
    draw = ImageDraw.Draw(img_pil)

    def draw_circle_outline(cx, cy, r, color):
        x0, y0 = cx - r, cy - r
        x1, y1 = cx + r, cy + r
        for w in range(line_width):
            draw.ellipse([x0 - w, y0 - w, x1 + w, y1 + w], outline=color)

    if gt_binary.any():
        cx, cy, r = circle_from_binary_mask(gt_binary)
        draw_circle_outline(cx, cy, r, color=(0, 200, 0))

    if pred_binary.any():
        cx, cy, r = circle_from_binary_mask(pred_binary)
        draw_circle_outline(cx, cy, r, color=(220, 0, 0))

    draw.text((10, 10), f"IoU: {iou:.3f}", fill=(255, 255, 255))
    return img_pil


def process_results_dir(results_dir):
    results_json_path = os.path.join(results_dir, "results.json")
    with open(results_json_path) as f:
        per_image_iou = json.load(f)["per_image_iou"]

    folders = sorted(
        f for f in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(f)
    )

    n_done = 0
    for folder in folders:
        folder_name = os.path.basename(folder)
        img_path = os.path.join(folder, "img_0000.png")
        gt_path = os.path.join(folder, "mask_0000.png")
        pred_path = os.path.join(folder, "saliency_mask_0000.png")
        if not (os.path.exists(img_path) and os.path.exists(gt_path) and os.path.exists(pred_path)):
            print(f"  [skip] {folder_name}: missing input file")
            continue
        if folder_name not in per_image_iou:
            print(f"  [skip] {folder_name}: not in results.json")
            continue

        img_np = np.array(Image.open(img_path).convert("RGB"))
        gt_binary = (np.array(Image.open(gt_path).convert("L")) >= 128).astype(np.uint8)
        pred_binary = (np.array(Image.open(pred_path).convert("L")) > 0).astype(np.uint8)
        iou = float(per_image_iou[folder_name])

        overlay = draw_iou_overlay(img_np, pred_binary, gt_binary, iou)
        overlay.save(os.path.join(folder, "iou_overlay_0000.png"))
        n_done += 1

    print(f"  Wrote {n_done} overlays in {results_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=ROOT_DIR)
    args = parser.parse_args()

    for sub in ("random", "saliency"):
        d = os.path.join(args.root_dir, sub)
        if os.path.isdir(d):
            print(f"Processing {d}")
            process_results_dir(d)
        else:
            print(f"  [skip] {d} does not exist")


if __name__ == "__main__":
    main()
