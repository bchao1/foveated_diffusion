"""Pre-compute and cache the CLIP score for a directory of real images,
using the same preprocessing as generate_metrics.py, so that metric runs can
load clipsim_real from gt_clip.json instead of re-processing real images."""
import logging
import os
import json
import sys
import pandas as pd
import torch
import torch.nn.functional as F
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_metrics import GEN_H, GEN_W, crop_and_resize, make_per_image_foveation_transform
from metric_computation.CLIP import _load_openclip, _clip_sim_batch_loop

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Pre-compute CLIP score for real images and save as gt_clip.json."
    )
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory of real images. Must contain metadata_test.csv with 'image' and 'prompt' columns.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where gt_clip.json is written.")
    parser.add_argument("--foveate_real", action="store_true",
                        help="Apply per-image foveation to real images. Images without a mask use center-square fallback.")
    parser.add_argument("--masks_dir", type=str, default=None,
                        help="Directory of mask_*.png files matched positionally to sorted images in images_dir. Optional; missing masks use center-square fallback.")
    parser.add_argument("--downsample_real", action="store_true",
                        help="Bicubic 2x downsample images before CLIP feature extraction.")
    parser.add_argument("--device", type=str, default=None,
                        help="Comma-separated PyTorch device strings (e.g. 'cuda:0,cuda:1'). Primary device is used. Defaults to cuda:0 if available.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for CLIP feature extraction (default: 64)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    _raw_device = args.device if args.device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = [d.strip() for d in _raw_device.split(',')][0]

    if not os.path.exists(args.images_dir):
        raise FileNotFoundError(f"images_dir not found: {args.images_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load image paths and prompts from metadata_test.csv.
    test_metadata_path = os.path.join(args.images_dir, "metadata_test.csv")
    if not os.path.exists(test_metadata_path):
        raise FileNotFoundError(f"Test metadata not found: {test_metadata_path}")
    test_metadata = pd.read_csv(test_metadata_path)

    def _norm_csv_prompt(s):
        s = s.strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s

    image_paths = [os.path.join(args.images_dir, p) for p in test_metadata["image"]]
    prompts     = [_norm_csv_prompt(p) for p in test_metadata["prompt"]]
    print(f"Found {len(image_paths)} images in metadata_test.csv")

    # Build preprocessing transforms (same as generate_metrics.py / extract_inception_features.py).
    pil_transform   = lambda img: crop_and_resize(img, GEN_H, GEN_W)
    image_transform = None

    if args.foveate_real:
        if args.masks_dir is None:
            print("WARNING: --masks_dir not provided with --foveate_real. Using center square mask for all images.")
            mask_paths = [None] * len(image_paths)
        else:
            mask_files = sorted(
                os.path.join(args.masks_dir, f)
                for f in os.listdir(args.masks_dir)
                if f.startswith('mask_') and f.endswith('.png')
            )
            mask_paths = [mp if os.path.isfile(mp) else None for mp in mask_files[:len(image_paths)]]
            if len(mask_paths) < len(image_paths):
                n_missing = len(image_paths) - len(mask_paths)
                print(f"WARNING: {n_missing} image(s) have no mask in masks_dir. Using center square mask.")
                mask_paths += [None] * n_missing
        # CLIP uses [0, 1] range → vmin=0, vmax=1
        image_transform = make_per_image_foveation_transform(mask_paths, device, vmin=0, vmax=1)

    elif args.downsample_real:
        def image_transform(batch):
            H, W = batch.shape[2], batch.shape[3]
            return F.interpolate(
                batch, size=(H // 2, W // 2),
                mode='bicubic', align_corners=False, antialias=True,
            ).clamp(0.0, 1.0)

    # Load CLIP model.
    model, tokenizer = _load_openclip(device=device)
    model.eval()

    image_size = model.visual.image_size
    if isinstance(image_size, tuple):
        image_size = image_size[0]
    image_size = int(image_size)
    image_mean = torch.tensor(model.visual.image_mean, device=device).view(3, 1, 1)
    image_std  = torch.tensor(model.visual.image_std,  device=device).view(3, 1, 1)
    dtype      = torch.float16

    # Compute CLIP score.
    clipsim_real = _clip_sim_batch_loop(
        image_paths, prompts, model, tokenizer,
        image_size, image_mean, image_std, dtype,
        args.batch_size, device,
        pil_transform=pil_transform,
        image_transform=image_transform,
    )
    print(f"CLIP Sim (real): {clipsim_real:.4f}")

    # Save.
    out_path = os.path.join(args.output_dir, "gt_clip.json")
    with open(out_path, "w") as f:
        json.dump({"clipsim_real": clipsim_real}, f, indent=2)
    print(f"gt_clip.json saved to {out_path}")
