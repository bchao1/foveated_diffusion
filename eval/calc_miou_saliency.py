"""
Compute mIoU between saliency-predicted masks and ground-truth masks.

For each img_XXX folder in the target directory:
  - Load img_0000.png, run DeepGaze-IIE to find the saliency centroid,
    then place a circular mask of fixed radius _MASK_RADIUS (matching
    --mask_radius 0.3 in saliency_generation.sh) at that centroid.
  - Save the predicted binary mask as saliency_mask_0000.png
  - Binarize the ground-truth mask_0000.png at threshold 128
  - Compute IoU between predicted and GT masks
  - Save per-image IoU and mean IoU to results.json in the same directory

Usage:
    conda run -n imp_tome python eval/calc_miou_saliency.py
"""

import os
import json
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
import deepgaze_pytorch

# --- Saliency constants (must match flux2_image_foveated.py) ---
_INFERENCE_SCALE = 0.5
_SPATIAL_SIGMA = 20.0
_DENSE_THRESHOLD = 0.3
# Fixed mask radius matching --mask_radius in saliency_generation.sh
_MASK_RADIUS = 0.3

RESULTS_DIR = "/home/brianchc/foveated_diffusion/results/eccv2026_rebuttal/saliency_IoU_experiment/random"


def load_saliency_detector(device):
    detector = deepgaze_pytorch.DeepGazeIIE(pretrained=True)
    detector.eval()
    detector = detector.to(device)
    return detector


def predict_saliency_mask(detector, img_rgb_np, device):
    """
    img_rgb_np: H x W x 3 uint8 numpy array.
    Returns a binary (0/1) numpy mask at the original image resolution.
    Mirrors the saliency branch of get_foveation_mask_from_image exactly.
    """
    h_orig, w_orig = img_rgb_np.shape[:2]

    # Convert to [0, 255] float tensor (1, 3, H, W)
    img_255 = torch.from_numpy(img_rgb_np).float().permute(2, 0, 1).unsqueeze(0).to(device)

    h_inf = int(h_orig * _INFERENCE_SCALE)
    w_inf = int(w_orig * _INFERENCE_SCALE)
    img_inf = F.interpolate(img_255, size=(h_inf, w_inf), mode="bilinear", align_corners=False)
    centerbias = torch.zeros(1, h_inf, w_inf, device=device, dtype=torch.float32)

    with torch.no_grad():
        log_density = detector(img_inf, centerbias)

    smap = torch.exp(log_density[0, 0]).cpu().numpy()
    smax = float(smap.max())
    if smax > 0:
        smap = smap / smax

    smoothed = gaussian_filter(smap, sigma=_SPATIAL_SIGMA)
    smax = float(smoothed.max())
    if smax > 0:
        smoothed = smoothed / smax

    # Find saliency centroid from the dense (above-threshold) region
    h, w = smoothed.shape
    eps = 1e-8
    peak = float(smoothed.max())
    if peak < eps:
        cx_norm, cy_norm = 0.0, 0.0
    else:
        thresh = _DENSE_THRESHOLD * peak
        dense_mask = smoothed >= thresh
        dense_smap = smoothed * dense_mask
        dense_total = dense_smap.sum() + eps
        y_coords = np.arange(h, dtype=np.float64)
        x_coords = np.arange(w, dtype=np.float64)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        cx_px = (dense_smap * x_grid).sum() / dense_total
        cy_px = (dense_smap * y_grid).sum() / dense_total
        cx_norm = (cx_px / w) - 0.5
        cy_norm = (cy_px / h) - 0.5

    # Build binary circular mask at full image resolution with fixed radius
    cx = (cx_norm + 0.5) * w_orig
    cy = (cy_norm + 0.5) * h_orig
    diagonal = (h_orig ** 2 + w_orig ** 2) ** 0.5
    radius_px = _MASK_RADIUS * (diagonal / 2.0)
    y = np.arange(h_orig, dtype=np.float64)
    x = np.arange(w_orig, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    mask_binary = (dist_sq <= radius_px ** 2).astype(np.uint8)
    return mask_binary


def compute_iou(pred_mask, gt_mask):
    """Binary IoU. Both masks are boolean or 0/1 uint8 numpy arrays."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def circle_from_binary_mask(binary_mask):
    """Return (cx, cy, radius) in pixel coords from a filled binary circle mask."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cx = (cmin + cmax) / 2.0
    cy = (rmin + rmax) / 2.0
    radius = ((rmax - rmin) + (cmax - cmin)) / 4.0
    return cx, cy, radius


def draw_iou_overlay(img_np, pred_binary, gt_binary, iou, line_width=4):
    """
    Draw two colored circle outlines on the image:
      - Green: ground-truth foveation mask border
      - Red:   saliency-predicted mask border
    Also annotates IoU in the top-left corner.
    Returns a PIL Image.
    """
    img_pil = Image.fromarray(img_np).convert("RGB")
    draw = ImageDraw.Draw(img_pil)

    def draw_circle_outline(cx, cy, r, color):
        x0, y0 = cx - r, cy - r
        x1, y1 = cx + r, cy + r
        for w in range(line_width):
            draw.ellipse([x0 - w, y0 - w, x1 + w, y1 + w], outline=color)

    if gt_binary.any():
        cx, cy, r = circle_from_binary_mask(gt_binary)
        draw_circle_outline(cx, cy, r, color=(0, 200, 0))   # green = GT

    if pred_binary.any():
        cx, cy, r = circle_from_binary_mask(pred_binary)
        draw_circle_outline(cx, cy, r, color=(220, 0, 0))   # red = predicted

    draw.text((10, 10), f"IoU: {iou:.3f}", fill=(255, 255, 255))

    return img_pil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading DeepGaze-IIE saliency detector...")
    detector = load_saliency_detector(device)

    folders = sorted(glob.glob(os.path.join(args.results_dir, "*")))
    folders = [f for f in folders if os.path.isdir(f)]
    print(f"Found {len(folders)} folders in {args.results_dir}")

    per_image_iou = {}

    for folder in folders:
        folder_name = os.path.basename(folder)
        img_path = os.path.join(folder, "img_0000.png")
        mask_path = os.path.join(folder, "mask_0000.png")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"  Skipping {folder_name}: missing img or mask")
            continue

        img_np = np.array(Image.open(img_path).convert("RGB"))
        gt_mask_np = np.array(Image.open(mask_path).convert("L"))
        gt_binary = (gt_mask_np >= 128).astype(np.uint8)

        pred_binary = predict_saliency_mask(detector, img_np, device)

        # Save predicted mask
        out_mask_path = os.path.join(folder, "saliency_mask_0000.png")
        Image.fromarray((pred_binary * 255).astype(np.uint8)).save(out_mask_path)

        iou = compute_iou(pred_binary, gt_binary)
        per_image_iou[folder_name] = iou
        print(f"  {folder_name}: IoU = {iou:.4f}")

        # Save overlay: image with GT (green) and predicted (red) circle borders
        overlay = draw_iou_overlay(img_np, pred_binary, gt_binary, iou)
        overlay.save(os.path.join(folder, "iou_overlay_0000.png"))

    if per_image_iou:
        mean_iou = float(np.mean(list(per_image_iou.values())))
    else:
        mean_iou = 0.0

    print(f"\nmIoU over {len(per_image_iou)} images: {mean_iou:.4f}")

    results = {"per_image_iou": per_image_iou, "mean_iou": mean_iou}
    out_json = os.path.join(args.results_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()
