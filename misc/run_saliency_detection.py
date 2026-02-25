"""
Run DeepGaze saliency detection on all images in a directory (recursive),
extract foveation parameters (center, radius) from saliency, generate foveation
masks, and save image, saliency map, mask, and foveated preview.
Uses the same logic as fovetaion_masks_from_video.py (single-frame version).
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw

# Optional: use diffsynth fov_utils if available (same as fovetaion_masks_from_video.py)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _path in [_REPO_ROOT / "DiffSynth-Studio", _REPO_ROOT]:
    _d = _path / "diffsynth"
    if _d.is_dir():
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
        break

try:
    from diffsynth.utils.fov_utils import create_foveation_mask_from_params, draw_foveation_circle_from_params
    _HAS_FOV_UTILS = True
except ImportError:
    _HAS_FOV_UTILS = False

import deepgaze_pytorch


# Same constants as fovetaion_masks_from_video.py
INFERENCE_SCALE = 0.5
SPATIAL_SIGMA = 20.0
R_MIN, R_MAX = 0.2, 0.5
COVERAGE = 0.9
DENSE_THRESHOLD = 0.3
MASK_SIGMA = 10.0

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_deepgaze_model(device):
    """Load DeepGaze IIE model with pretrained weights (same as video script)."""
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
    model.eval()
    return model


def _frame_to_tensor(pil_img):
    """Convert a PIL Image to float tensor [1, 3, H, W] in [0, 255] (same as video script)."""
    arr = np.array(pil_img).astype(np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _saliency_to_params(smap, r_min=R_MIN, r_max=R_MAX, coverage=COVERAGE, dense_threshold=DENSE_THRESHOLD):
    """
    Extract (cx, cy, r) from a single (smoothed) saliency heatmap using weighted centroid
    over the dense core. Same as fovetaion_masks_from_video.py.
    """
    h, w = smap.shape
    eps = 1e-8
    peak = smap.max()
    if peak < eps:
        return (0.0, 0.0), r_min

    thresh = dense_threshold * peak
    dense_mask = smap >= thresh
    dense_smap = smap * dense_mask
    dense_total = dense_smap.sum() + eps

    y_coords = np.arange(h, dtype=np.float64)
    x_coords = np.arange(w, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    cx_px = (dense_smap * x_grid).sum() / dense_total
    cy_px = (dense_smap * y_grid).sum() / dense_total

    cx_norm = (cx_px / w) - 0.5
    cy_norm = (cy_px / h) - 0.5

    dist = np.sqrt((x_grid - cx_px) ** 2 + (y_grid - cy_px) ** 2)
    dense_pixels = dense_mask.ravel()
    flat_dist = dist.ravel()[dense_pixels]
    flat_sal = dense_smap.ravel()[dense_pixels]

    sort_idx = np.argsort(flat_dist)
    cumsum = np.cumsum(flat_sal[sort_idx])
    target = coverage * dense_total
    hit = np.searchsorted(cumsum, target)
    hit = min(hit, len(flat_dist) - 1)
    radius_px = flat_dist[sort_idx[hit]]

    half_diag = 0.5 * np.sqrt(h**2 + w**2)
    r = radius_px / half_diag
    r = float(np.clip(r, r_min, r_max))
    return (float(cx_norm), float(cy_norm)), r


def _create_foveation_mask_single_image(height, width, center, r, sigma=MASK_SIGMA):
    """
    Create a single full-res foveation mask (H, W) with optional Gaussian soft falloff.
    center: (cx_norm, cy_norm) in [-0.5, 0.5]; r: radius relative to half diagonal.
    """
    cx = (center[0] + 0.5) * width
    cy = (center[1] + 0.5) * height
    diagonal = (height**2 + width**2) ** 0.5
    radius_px = r * (diagonal / 2.0)

    y = np.arange(height, dtype=np.float64)
    x = np.arange(width, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (dist_sq <= radius_px**2).astype(np.float32)
    if sigma > 0:
        mask = gaussian_filter(mask, sigma=sigma)
        mask = mask / (mask.max() + 1e-8)
    return mask


def _draw_foveation_circle_on_image(pil_image, center, r):
    """Draw foveation circle on a PIL image. center (cx_norm, cy_norm), r relative to half diag."""
    w, h = pil_image.size
    cx = (center[0] + 0.5) * w
    cy = (center[1] + 0.5) * h
    diagonal = (w**2 + h**2) ** 0.5
    radius_px = r * (diagonal / 2.0)

    out = pil_image.copy()
    draw = ImageDraw.Draw(out)
    draw.ellipse(
        (cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px),
        outline=(255, 0, 0),
        width=max(2, int(radius_px * 0.05)),
    )
    return out


def predict_saliency_and_mask(model, pil_image, device):
    """
    Same pipeline as fovetaion_masks_from_video.predict_saliency_maps but for a single image:
    half-res inference -> spatial smoothing -> param extraction -> foveation mask.

    Returns:
        mask_np: (H_orig, W_orig) float mask in [0, 1].
        smoothed_smap: (h_inf, w_inf) numpy array for visualization.
        center: (cx_norm, cy_norm).
        r: float radius.
    """
    w_orig, h_orig = pil_image.size
    w_inf = int(w_orig * INFERENCE_SCALE)
    h_inf = int(h_orig * INFERENCE_SCALE)
    resized = pil_image.resize((w_inf, h_inf), Image.LANCZOS)

    centerbias = torch.zeros(1, h_inf, w_inf).to(device)
    img_tensor = _frame_to_tensor(resized).to(device)

    with torch.no_grad():
        log_density = model(img_tensor, centerbias)
    smap = np.exp(log_density[0, 0].cpu().numpy())
    smax = smap.max()
    if smax > 0:
        smap = smap / smax

    # Single image: only spatial smoothing (no temporal)
    smoothed_smap = gaussian_filter(smap, sigma=SPATIAL_SIGMA)
    smax = smoothed_smap.max()
    if smax > 0:
        smoothed_smap = smoothed_smap / smax

    center, r = _saliency_to_params(smoothed_smap)

    if _HAS_FOV_UTILS:
        masks = create_foveation_mask_from_params(
            keyframe_centers=[center],
            keyframe_radii=[r],
            num_frames=1,
            height=h_orig,
            width=w_orig,
            sigma=MASK_SIGMA,
            device=torch.device(device),
            dtype=torch.float32,
        )
        mask_np = masks[0, 0, 0].cpu().numpy()
    else:
        mask_np = _create_foveation_mask_single_image(h_orig, w_orig, center, r, sigma=MASK_SIGMA)

    return mask_np, smoothed_smap, center, r


def collect_image_paths(root_dir):
    """Recursively collect all image paths under root_dir."""
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Images directory not found: {root_dir}")
    paths = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return sorted(paths)


def save_saliency_heatmap(saliency_np, out_path, size_orig=None):
    """Save saliency as a heatmap PNG. If size_orig, upscale to match original image size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if size_orig is not None:
        h_orig, w_orig = size_orig
        saliency_np = np.array(
            Image.fromarray((saliency_np * 255).astype(np.uint8)).resize(
                (w_orig, h_orig), Image.BILINEAR
            )
        ).astype(np.float32) / 255.0

    plt.figure(figsize=(saliency_np.shape[1] / 100, saliency_np.shape[0] / 100), dpi=100)
    plt.imshow(saliency_np, cmap="hot", vmin=0, vmax=1)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run saliency detection and foveation mask generation on images (same logic as fovetaion_masks_from_video)."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/local/brian/Aesthetic-Train-V2/",
        help="Root directory containing images (searched recursively)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/brianchc/foveated_diffusion/results/saliency_detection/",
        help="Output directory for images, saliency maps, and masks",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1024,
        help="Max side length for input images (resize for speed). Default: no resize",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_foveated_preview", action="store_true", help="Save foveated image preview (mask * img + (1-mask) * img_lr)")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    print("Loading DeepGaze IIE saliency model...")
    model = load_deepgaze_model(device)

    image_paths = collect_image_paths(images_dir)
    print(f"Found {len(image_paths)} images under {images_dir}")

    for i, path in enumerate(image_paths):
        rel = path.relative_to(images_dir)
        out_sub = output_dir / rel.parent
        out_sub.mkdir(parents=True, exist_ok=True)
        stem = path.stem
        out_image_path = out_sub / f"{stem}_image{path.suffix}"
        out_saliency_path = out_sub / f"{stem}_saliency.png"
        out_mask_path = out_sub / f"{stem}_foveation_mask.png"
        out_foveated_path = out_sub / f"{stem}_foveated_preview{path.suffix}"

        if out_saliency_path.exists() and out_mask_path.exists():
            print(f"[{i+1}/{len(image_paths)}] Skip (exists): {rel}")
            continue

        try:
            pil_image = Image.open(path).convert("RGB")
            w_orig, h_orig = pil_image.size

            if args.max_size is not None and max(w_orig, h_orig) > args.max_size:
                scale = args.max_size / max(w_orig, h_orig)
                new_w, new_h = int(round(w_orig * scale)), int(round(h_orig * scale))
                pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                w_orig, h_orig = new_w, new_h

            mask_np, smoothed_smap, center, r = predict_saliency_and_mask(model, pil_image, device)

            # Save copy of input image
            pil_image.save(out_image_path)

            # Save saliency heatmap (upscaled to original size for consistency)
            save_saliency_heatmap(smoothed_smap, out_saliency_path, size_orig=(h_orig, w_orig))

            # Save foveation mask as grayscale PNG
            mask_uint8 = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(mask_uint8).save(out_mask_path)

            if args.save_foveated_preview:
                img_tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                img_lr = F.interpolate(img_tensor, scale_factor=0.5, mode="bilinear", align_corners=False)
                img_lr_up = F.interpolate(img_lr, scale_factor=2.0, mode="bilinear", align_corners=False)
                fov_img = mask_tensor * img_tensor + (1 - mask_tensor) * img_lr_up
                fov_np = (fov_img[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(fov_np).save(out_foveated_path)

            # Save image with foveation circle drawn
            circle_path = out_sub / f"{stem}_circle{path.suffix}"
            if _HAS_FOV_UTILS:
                img_np = np.array(pil_image)
                frames_with_circle = draw_foveation_circle_from_params([img_np], [center], [r])
                Image.fromarray(frames_with_circle[0]).save(circle_path)
            else:
                _draw_foveation_circle_on_image(pil_image, center, r).save(circle_path)

            print(f"[{i+1}/{len(image_paths)}] {rel} -> saliency, mask, circle")
        except Exception as e:
            import traceback
            print(f"[{i+1}/{len(image_paths)}] Error {path}: {e}")
            traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    main()
