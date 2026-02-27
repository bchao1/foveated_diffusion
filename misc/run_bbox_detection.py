"""
Run multi-object bounding box detection on all images in a directory (recursive),
generate a foveation mask based on detected boxes, and save image, mask, and
foveated preview.

This follows the overall structure and CLI style of run_saliency_detection.py
but uses a fast, state-of-the-art YOLO detector instead of saliency.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Optional: use diffsynth fov_utils if available (for consistency with saliency script)
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


def load_yolo_detector(model_name: str, device: str = "cuda"):
    """
    Load a YOLO detector from the ultralytics package.

    We keep this minimal and defer to ultralytics for choosing the best
    available backbone; the default model is a good speed/accuracy trade-off.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics not installed. Please install with `pip install ultralytics`."
        ) from exc

    model = YOLO(model_name)
    # Device is specified at inference time; this keeps compatibility across versions.
    model.model.float()
    return model


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


def _create_foveation_mask_from_bboxes(
    height: int,
    width: int,
    bboxes_xyxy,
    sigma: float = 10.0,
):
    """
    Create a full-resolution foveation mask (H, W) in [0, 1] from a list of
    bounding boxes in absolute pixel coordinates (x1, y1, x2, y2).

    The mask is 1 inside any bbox, 0 outside, followed by Gaussian smoothing
    and normalization. This yields a smooth "high-acuity" region covering all
    detected objects.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    for box in bboxes_xyxy:
        x1, y1, x2, y2 = box
        x1_i = max(0, min(width - 1, int(np.floor(x1))))
        y1_i = max(0, min(height - 1, int(np.floor(y1))))
        x2_i = max(0, min(width, int(np.ceil(x2))))
        y2_i = max(0, min(height, int(np.ceil(y2))))
        if x2_i <= x1_i or y2_i <= y1_i:
            continue
        mask[y1_i:y2_i, x1_i:x2_i] = 1.0

    if sigma > 0:
        mask = gaussian_filter(mask, sigma=sigma)
    max_val = mask.max()
    if max_val > 0:
        mask = mask / max_val
    return mask


def _bboxes_to_circle_params(height: int, width: int, bboxes_xyxy):
    """
    Approximate each bbox as a circular high-acuity region, returning
    (centers, radii) in the same normalized coordinates as the saliency script:
    - center: (cx_norm, cy_norm) in [-0.5, 0.5]
    - r: radius relative to half of the image diagonal
    """
    if len(bboxes_xyxy) == 0:
        return [], []

    diagonal = float((height**2 + width**2) ** 0.5)
    centers = []
    radii = []
    for x1, y1, x2, y2 in bboxes_xyxy:
        cx_px = 0.5 * (x1 + x2)
        cy_px = 0.5 * (y1 + y2)
        w_box = max(0.0, x2 - x1)
        h_box = max(0.0, y2 - y1)
        r_px = 0.5 * float((w_box**2 + h_box**2) ** 0.5)

        cx_norm = (cx_px / width) - 0.5
        cy_norm = (cy_px / height) - 0.5
        r_norm = r_px / (0.5 * diagonal + 1e-8)

        centers.append((float(cx_norm), float(cy_norm)))
        radii.append(float(r_norm))
    return centers, radii


def draw_bboxes_on_image(pil_image: Image.Image, bboxes_xyxy, scores=None, class_ids=None):
    """Draw bounding boxes (and optional scores / class IDs) on a copy of the image."""
    out = pil_image.copy()
    draw = ImageDraw.Draw(out)
    for i, box in enumerate(bboxes_xyxy):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
        label = None
        if scores is not None:
            label = f"{scores[i]:.2f}"
        if class_ids is not None:
            if label is None:
                label = str(class_ids[i])
            else:
                label = f"{class_ids[i]} {label}"
        if label is not None:
            text_pos = (x1 + 2, y1 + 2)
            draw.text(text_pos, label, fill=(255, 255, 255))
    return out


def stack_bbox_mask_image(
    overlay_img: Image.Image,
    mask_np: np.ndarray,
    orig_img: Image.Image,
) -> Image.Image:
    """
    Create a single stacked visualization image: (bbox overlay, foveation mask, original image).
    Stacking is horizontal, all at the same height/width.
    """
    w, h = orig_img.size
    # Ensure overlay matches size
    if overlay_img.size != (w, h):
        overlay_img = overlay_img.resize((w, h), Image.LANCZOS)

    # Convert mask to 3-channel grayscale image
    mask_uint8 = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_uint8).resize((w, h), Image.LANCZOS).convert("RGB")

    stacked = Image.new("RGB", (w * 3, h))
    stacked.paste(overlay_img, (0, 0))
    stacked.paste(mask_img, (w, 0))
    stacked.paste(orig_img, (2 * w, 0))
    return stacked


def run_detection_on_image(detector, pil_image: Image.Image, device: str, conf_thres: float = 0.25):
    """
    Run YOLO detection on a single PIL image and return:
        - bboxes_xyxy: list of (x1, y1, x2, y2) in pixel space
        - scores: list of confidences
        - class_ids: list of class indices
    """
    img_np = np.array(pil_image)  # RGB

    # ultralytics API: model.predict / model(...)
    results = detector.predict(
        img_np,
        conf=conf_thres,
        device=device,
        verbose=False,
    )
    if not results:
        return [], [], []

    res = results[0]
    if res.boxes is None or res.boxes.xyxy is None or len(res.boxes) == 0:
        return [], [], []

    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    class_ids = res.boxes.cls.cpu().numpy().astype(int)

    bboxes_xyxy = [tuple(map(float, box)) for box in boxes]
    scores = [float(s) for s in scores]
    class_ids = [int(c) for c in class_ids]
    return bboxes_xyxy, scores, class_ids


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO multi-object detection and foveation mask generation on images."
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
        default="/home/brianchc/foveated_diffusion/results/bbox_detection/",
        help="Output directory for images, masks, and previews",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1024,
        help="Max side length for input images (resize for speed).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device / YOLO device to use.",
    )
    parser.add_argument(
        "--detector_model",
        type=str,
        default="yolov8n.pt",
        help=(
            "YOLO model name or path. "
            "Defaults to a fast, small YOLOv8 model; "
            "you can use yolov8s.pt / yolov8m.pt / yolov8x.pt, etc."
        ),
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=10.0,
        help="Gaussian sigma for smoothing the bbox-based foveation mask.",
    )
    parser.add_argument(
        "--stack_outputs",
        action="store_true",
        help="Save a single stacked image (bbox overlay | foveation mask | image) instead of separate bbox/mask images.",
    )
    parser.add_argument(
        "--save_foveated_preview",
        action="store_true",
        help="Save foveated image preview (mask * img + (1-mask) * img_lr).",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() or "cpu" not in args.device.lower() else "cpu"

    print(f"Loading YOLO detector '{args.detector_model}' on device '{device}'...")
    detector = load_yolo_detector(args.detector_model, device=device)

    image_paths = collect_image_paths(images_dir)
    print(f"Found {len(image_paths)} images under {images_dir}")

    for i, path in enumerate(image_paths):
        rel = path.relative_to(images_dir)
        out_sub = output_dir / rel.parent
        out_sub.mkdir(parents=True, exist_ok=True)

        stem = path.stem
        out_image_path = out_sub / f"{stem}_image{path.suffix}"
        out_mask_path = out_sub / f"{stem}_foveation_mask.png"
        out_overlay_path = out_sub / f"{stem}_bboxes{path.suffix}"
        out_stacked_path = out_sub / f"{stem}_stacked.png"
        out_foveated_path = out_sub / f"{stem}_foveated_preview{path.suffix}"

        if args.stack_outputs:
            exists_flag = out_stacked_path.exists()
        else:
            exists_flag = out_mask_path.exists() and out_overlay_path.exists()

        if exists_flag:
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

            bboxes_xyxy, scores, class_ids = run_detection_on_image(
                detector, pil_image, device=device, conf_thres=args.conf_thres
            )

            if len(bboxes_xyxy) == 0:
                print(f"[{i+1}/{len(image_paths)}] {rel} -> no detections, skipping mask.")
                # Still save resized image to keep outputs aligned.
                pil_image.save(out_image_path)
                continue

            # Save copy of (potentially resized) input image.
            pil_image.save(out_image_path)

            # Build foveation mask from boxes.
            if _HAS_FOV_UTILS:
                centers, radii = _bboxes_to_circle_params(h_orig, w_orig, bboxes_xyxy)
                if len(centers) == 0:
                    mask_np = np.zeros((h_orig, w_orig), dtype=np.float32)
                else:
                    masks = create_foveation_mask_from_params(
                        keyframe_centers=centers,
                        keyframe_radii=radii,
                        num_frames=1,
                        height=h_orig,
                        width=w_orig,
                        sigma=args.sigma,
                        device=torch.device(device if torch.cuda.is_available() else "cpu"),
                        dtype=torch.float32,
                    )
                    # fuse multiple channels/regions by max over keyframes
                    mask_tensor = masks[0, 0]  # (K, H, W) or (1, H, W) depending on implementation
                    if mask_tensor.ndim == 3:
                        mask_np = torch.max(mask_tensor, dim=0).values.cpu().numpy()
                    else:
                        mask_np = mask_tensor[0].cpu().numpy()
            else:
                mask_np = _create_foveation_mask_from_bboxes(h_orig, w_orig, bboxes_xyxy, sigma=args.sigma)

            # Save image with bounding boxes overlaid.
            overlay_img = draw_bboxes_on_image(pil_image, bboxes_xyxy, scores=scores, class_ids=class_ids)

            if args.stack_outputs:
                # Save a single stacked visualization: [overlay | mask | image]
                stacked = stack_bbox_mask_image(overlay_img, mask_np, pil_image)
                stacked.save(out_stacked_path)
            else:
                # Save foveation mask as grayscale PNG.
                mask_uint8 = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(mask_uint8).save(out_mask_path)

                # Save overlay image separately.
                overlay_img.save(out_overlay_path)

            # Optional: foveated preview similar to saliency script.
            if args.save_foveated_preview:
                img_tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                img_lr = F.interpolate(img_tensor, scale_factor=0.5, mode="bilinear", align_corners=False)
                img_lr_up = F.interpolate(img_lr, scale_factor=2.0, mode="bilinear", align_corners=False)
                fov_img = mask_tensor * img_tensor + (1 - mask_tensor) * img_lr_up
                fov_np = (fov_img[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(fov_np).save(out_foveated_path)

            if args.stack_outputs:
                print(f"[{i+1}/{len(image_paths)}] {rel} -> stacked (bboxes, mask, image)")
            else:
                print(f"[{i+1}/{len(image_paths)}] {rel} -> bboxes, mask, overlay")
        except Exception as e:
            import traceback

            print(f"[{i+1}/{len(image_paths)}] Error {path}: {e}")
            traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    main()

