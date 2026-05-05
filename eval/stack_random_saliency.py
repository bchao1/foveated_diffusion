"""
Stack random/saliency iou_overlay images side-by-side for all matched pairs.

For each matching folder ID in:
  saliency_IoU_experiment/random/<ID>/iou_overlay_0000.png  (left)
  saliency_IoU_experiment/saliency/<ID>/iou_overlay_0000.png  (right)

Produce a side-by-side image with 1% of total width as white padding between them,
saved to:
  /home/brianchc/foveated_diffusion/results/eccv2026_rebuttal/stacked/<ID>.png

Usage:
    conda run -n imp_tome python eval/stack_random_saliency.py
"""

import os
import glob

import numpy as np
from PIL import Image

EXPERIMENT_DIR = "/home/brianchc/foveated_diffusion/results/eccv2026_rebuttal/saliency_IoU_experiment"
OUT_DIR = "/home/brianchc/foveated_diffusion/results/eccv2026_rebuttal/stacked"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    random_dirs = {os.path.basename(d): d for d in sorted(glob.glob(os.path.join(EXPERIMENT_DIR, "random", "*"))) if os.path.isdir(d)}
    saliency_dirs = {os.path.basename(d): d for d in sorted(glob.glob(os.path.join(EXPERIMENT_DIR, "saliency", "*"))) if os.path.isdir(d)}

    ids = sorted(set(random_dirs) & set(saliency_dirs))
    print(f"Found {len(ids)} matched pairs")

    for idx in ids:
        rand_path = os.path.join(random_dirs[idx], "iou_overlay_0000.png")
        sal_path = os.path.join(saliency_dirs[idx], "iou_overlay_0000.png")

        if not os.path.exists(rand_path) or not os.path.exists(sal_path):
            print(f"  Skipping {idx}: missing overlay")
            continue

        left = np.array(Image.open(rand_path).convert("RGB"))
        right = np.array(Image.open(sal_path).convert("RGB"))

        h = max(left.shape[0], right.shape[0])
        # Pad shorter image with white on bottom if heights differ
        def pad_h(img, target_h):
            if img.shape[0] < target_h:
                pad = np.full((target_h - img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
                img = np.concatenate([img, pad], axis=0)
            return img

        left = pad_h(left, h)
        right = pad_h(right, h)

        total_w = left.shape[1] + right.shape[1]
        gap = max(1, round(0.01 * total_w))
        white_gap = np.full((h, gap, 3), 255, dtype=np.uint8)

        stacked = np.concatenate([left, white_gap, right], axis=1)
        out_path = os.path.join(OUT_DIR, f"{idx}.png")
        Image.fromarray(stacked).save(out_path)

    print(f"Saved {len(ids)} stacked images to {OUT_DIR}")


if __name__ == "__main__":
    main()
