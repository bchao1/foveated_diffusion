"""
Utility to draw foveation outlines on a folder of images, using the
same foveation mask and outline logic as in eval/inference.py.
"""

import os
from typing import Tuple, Union

import numpy as np
from PIL import Image
import torch

# Reuse the exact same functions as inference.py
from inference import create_foveation_mask_full_res, draw_foveation_outline


def draw_outlines_for_folder(
    input_dir: str,
    output_dir: str,
    height: int = 1024,
    width: int = 1024,
    mask_radius: float = 0.5,
    full_eval_mask: str = "square",
    outline_width_frac: float = 0.01,
    outline_color: Union[str, Tuple[int, int, int]] = "255,0,0",
    center: Tuple[float, float] = (0.0, 0.0),
) -> None:
    """
    Draw foveation outlines on all images in a folder.

    Uses the same full-resolution foveation mask definition as in
    eval/inference.py:
      - create_foveation_mask_full_res(height, width, center, r, shape)
      - draw_foveation_outline(...)

    Args:
        input_dir: Folder containing input images.
        output_dir: Folder to write outlined images (created if needed).
        height, width: Full image size for the foveation mask (e.g. 1024x1024).
        mask_radius: Radius r as in inference.py (relative to half-diagonal
            for circular, or ratio 0–1 for square).
        full_eval_mask: Mask shape, same as args.full_eval_mask /
            create_foveation_mask{_full_res} `shape` argument
            (e.g. \"circular\", \"square\").
        outline_width_frac: Border width as fraction of image width
            (same as --outline_width_frac in inference.py).
        outline_color: Outline color as \"R,G,B\" string or (R, G, B) tuple
            (same semantics as --outline_color in inference.py).
        center: Foveation center in normalized coords (-0.5, 0.5).
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(outline_color, str):
        outline_color_parsed = tuple(int(x) for x in outline_color.split(","))
    else:
        outline_color_parsed = outline_color

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Full-resolution mask, same function/signature as in inference.py
    full_res_mask = create_foveation_mask_full_res(
        height,
        width,
        center=center,
        r=mask_radius,
        shape=full_eval_mask,
        device=device,
    )

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith(exts):
            continue
        in_path = os.path.join(input_dir, name)
        out_path = os.path.join(output_dir, name)

        img = Image.open(in_path).convert("RGB")
        img_np = np.array(img)

        # Resize image to (height, width) so outline/mask align exactly
        if img_np.shape[0] != height or img_np.shape[1] != width:
            img = img.resize((width, height), Image.LANCZOS)
            img_np = np.array(img)

        outlined = draw_foveation_outline(
            img_np,
            full_res_mask,
            height,
            width,
            outline_width_frac=outline_width_frac,
            color=outline_color_parsed,
        )
        Image.fromarray(outlined).save(out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Draw foveation outlines on all images in a folder "
        "using the same mask/outline logic as eval/inference.py."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input image folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder for outlined images")
    parser.add_argument("--height", type=int, default=1024, help="Full image height (default: 1024)")
    parser.add_argument("--width", type=int, default=1024, help="Full image width (default: 1024)")
    parser.add_argument("--mask_radius", type=float, default=0.5, help="Mask radius r (same semantics as inference.py)")
    parser.add_argument(
        "--full_eval_mask",
        type=str,
        default="square",
        help="Mask shape (same as args.full_eval_mask / create_foveation_mask shape)",
    )
    parser.add_argument(
        "--outline_width_frac",
        type=float,
        default=0.01,
        help="Foveation outline width as fraction of image width",
    )
    parser.add_argument(
        "--outline_color",
        type=str,
        default="255,255,255",
        help="Foveation outline color as R,G,B (e.g. 255,0,0)",
    )
    parser.add_argument(
        "--center_x",
        type=float,
        default=0.0,
        help="Foveation center x in normalized coords (-0.5, 0.5)",
    )
    parser.add_argument(
        "--center_y",
        type=float,
        default=0.0,
        help="Foveation center y in normalized coords (-0.5, 0.5)",
    )

    args = parser.parse_args()
    draw_outlines_for_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        mask_radius=args.mask_radius,
        full_eval_mask=args.full_eval_mask,
        outline_width_frac=args.outline_width_frac,
        outline_color=args.outline_color,
        center=(args.center_x, args.center_y),
    )

