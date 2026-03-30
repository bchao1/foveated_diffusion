"""
Generate N random binary circular foveation masks for a single input image.
For each sample, saves:
  - binary mask
  - pixelated version (downsample to 64x64, upsample to 512x512, nearest)
  - input image with red circular border drawn

Uses the same visual style and constants as run_saliency_detection.py.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

R_MIN, R_MAX = 0.2, 0.5


def center_crop_square(pil_image):
    """Center-crop a PIL image to a square (min side)."""
    w, h = pil_image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return pil_image.crop((left, top, left + side, top + side))


def create_binary_circular_mask(height, width, center, r):
    """
    Create a binary circular mask (H, W) uint8 in {0, 255}.
    center: (cx_norm, cy_norm) in [-0.5, 0.5]; r: radius relative to half diagonal.
    """
    cx = (center[0] + 0.5) * width
    cy = (center[1] + 0.5) * height
    diagonal = (height ** 2 + width ** 2) ** 0.5
    radius_px = r * (diagonal / 2.0)

    y = np.arange(height, dtype=np.float64)
    x = np.arange(width, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (dist_sq <= radius_px ** 2).astype(np.uint8) * 255
    return mask


def draw_foveation_circle_on_image(pil_image, center, r):
    """Draw red foveation circle border on a PIL image."""
    w, h = pil_image.size
    cx = (center[0] + 0.5) * w
    cy = (center[1] + 0.5) * h
    diagonal = (w ** 2 + h ** 2) ** 0.5
    radius_px = r * (diagonal / 2.0)

    out = pil_image.copy()
    draw = ImageDraw.Draw(out)
    draw.ellipse(
        (cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px),
        outline=(255, 0, 0),
        width=max(2, int(radius_px * 0.05)),
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate N random binary circular foveation masks for a single image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/brianchc/foveated_diffusion/results/random_masks/",
        help="Output directory",
    )
    parser.add_argument("--n", type=int, default=10, help="Number of random samples")
    parser.add_argument(
        "--max_size",
        type=int,
        default=1024,
        help="Max side length for input image (resize for speed)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    image_path = Path(args.image)
    stem = image_path.stem
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save rainbow colormap strip (512x64, horizontal gradient, no text)
    import matplotlib.cm as cm
    gradient = np.linspace(0, 1, 512)[np.newaxis, :]  # (1, 512)
    strip_rgba = (cm.rainbow(gradient)[0] * 255).astype(np.uint8)  # (512, 4)
    strip_rgb = strip_rgba[:, :3]  # drop alpha -> (512, 3)
    strip_img = Image.fromarray(np.tile(strip_rgb, (64, 1, 1)))  # (64, 512, 3)
    strip_img.save(output_dir / "rainbow_colormap_strip.png")

    pil_image = Image.open(image_path).convert("RGB")
    pil_image = center_crop_square(pil_image)
    w, h = pil_image.size

    if args.max_size is not None and max(w, h) > args.max_size:
        scale = args.max_size / max(w, h)
        w, h = int(round(w * scale)), int(round(h * scale))
        pil_image = pil_image.resize((w, h), Image.LANCZOS)

    for i in range(args.n):
        cx_norm = float(np.random.uniform(-0.5, 0.5))
        cy_norm = float(np.random.uniform(-0.5, 0.5))
        r = float(np.random.uniform(R_MIN, R_MAX))
        center = (cx_norm, cy_norm)

        out_sub = output_dir / stem / f"{i:04d}"
        out_sub.mkdir(parents=True, exist_ok=True)

        # Binary mask
        mask = create_binary_circular_mask(h, w, center, r)
        Image.fromarray(mask).save(out_sub / "foveation_mask_binary.png")

        # Pixelated: downsample to 64x64, upsample to 512x512 (nearest)
        mask_pix = Image.fromarray(mask).resize((64, 64), Image.NEAREST).resize((512, 512), Image.NEAREST)
        mask_pix.save(out_sub / "foveation_mask_pix64_up512.png")

        # Image with red circle border
        circle_img = draw_foveation_circle_on_image(pil_image, center, r)
        circle_img.save(out_sub / f"circle{image_path.suffix}")

        print(f"[{i+1}/{args.n}] center=({cx_norm:.3f}, {cy_norm:.3f}) r={r:.3f} -> {out_sub}")

    print("Done.")


if __name__ == "__main__":
    main()
