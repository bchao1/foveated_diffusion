"""Pre-extract and cache Inception features for a directory of real or generated images,
using the same preprocessing as generate_metrics.py, so that metric runs can skip re-extraction."""
import logging
import os
import pickle
import sys
import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_metrics import GEN_H, GEN_W, crop_and_resize, make_per_image_foveation_transform
from metric_computation.inception_utils import (
    load_inception, extract_inception_features, _list_images,
)

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Pre-extract Inception features and save as a PKL cache compatible with generate_metrics.py."
    )
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory of images to process.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where the cache PKL is written.")
    parser.add_argument("--real", dest="real", action="store_true", default=True,
                        help="images_dir contains real images (default). Saves real_feats_cache.pkl.")
    parser.add_argument("--no-real", dest="real", action="store_false",
                        help="images_dir contains generated images. Saves gen_feats_cache.pkl.")
    parser.add_argument("--foveate_real", action="store_true",
                        help="Apply per-image foveation to real images (requires --masks_dir).")
    parser.add_argument("--masks_dir", type=str, default=None,
                        help="Directory of mask_*.png files matched positionally to sorted images in images_dir.")
    parser.add_argument("--downsample_real", action="store_true",
                        help="Bicubic 2x downsample images before feature extraction.")
    parser.add_argument("--device", type=str, default=None,
                        help="Comma-separated PyTorch device strings (e.g. 'cuda:0,cuda:1'). Multiple GPUs use DataParallel. Defaults to cuda:0 if available.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for Inception feature extraction (default: 64)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if args.foveate_real and args.real and args.masks_dir is None:
        parser.error("--masks_dir is required when --foveate_real and --real are both set.")

    _raw_device = args.device if args.device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    devices = [d.strip() for d in _raw_device.split(',')]
    device = devices[0]
    gpu_ids = [int(d.split(':')[1]) for d in devices if d.startswith('cuda:') and ':' in d]

    if not os.path.exists(args.images_dir):
        raise FileNotFoundError(f"images_dir not found: {args.images_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.real:
        test_metadata_path = os.path.join(args.images_dir, "metadata_test.csv")
        if not os.path.exists(test_metadata_path):
            raise FileNotFoundError(f"Test metadata not found: {test_metadata_path}")
        test_metadata = pd.read_csv(test_metadata_path)
        image_paths = [os.path.join(args.images_dir, p) for p in test_metadata["image"]]
    else:
        image_paths = _list_images(args.images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")
    print(f"Found {len(image_paths)} images in {args.images_dir}")

    # Build preprocessing transforms (same as generate_metrics.py).
    pil_transform    = None
    image_transform  = None

    if args.real:
        pil_transform = lambda img: crop_and_resize(img, GEN_H, GEN_W)

        if args.foveate_real:
            # Build mask path list by positional matching with sorted masks in masks_dir.
            mask_files = sorted(
                os.path.join(args.masks_dir, f)
                for f in os.listdir(args.masks_dir)
                if f.startswith('mask_') and f.endswith('.png')
            )
            if len(mask_files) != len(image_paths):
                raise ValueError(
                    f"Number of masks ({len(mask_files)}) != number of images ({len(image_paths)}). "
                    "Ensure masks_dir has one mask per image in sorted order."
                )
            # Validate mask paths; missing files become None (center-square fallback).
            mask_paths = []
            for mp in mask_files:
                if os.path.isfile(mp):
                    mask_paths.append(mp)
                else:
                    print(f"WARNING: mask not found: {mp}. Using center square mask.")
                    mask_paths.append(None)
            image_transform = make_per_image_foveation_transform(mask_paths, device, vmin=-1, vmax=1)

        elif args.downsample_real:
            import torch.nn.functional as F
            def image_transform(batch):
                H, W = batch.shape[2], batch.shape[3]
                return F.interpolate(
                    batch, size=(H // 2, W // 2),
                    mode='bicubic', align_corners=False, antialias=True,
                ).clamp(-1.0, 1.0)

    # Load Inception model and extract features.
    inception_net, inception_style = load_inception(
        {'type': 'StyleGAN'}, metric='FID', map_location='cpu')
    inception_net = inception_net.to(device).eval()
    if len(gpu_ids) > 1:
        inception_net = torch.nn.DataParallel(inception_net, device_ids=gpu_ids)

    feats, preds = extract_inception_features(
        image_paths, inception_net, inception_style,
        batch_size=args.batch_size, device=device, resize=False,
        image_transform=image_transform,
        pil_transform=pil_transform,
    )

    # Save cache in the format expected by generate_metrics.py.
    if args.real:
        out_path = os.path.join(args.output_dir, "real_feats_cache.pkl")
        real_mean = np.mean(feats, axis=0)
        real_cov  = np.cov(feats, rowvar=False)
        with open(out_path, 'wb') as f:
            pickle.dump(dict(mean=real_mean, cov=real_cov,
                             real_feats_np=feats,
                             num_real_feeded=len(image_paths)), f)
    else:
        out_path = os.path.join(args.output_dir, "gen_feats_cache.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(dict(feats_np=feats, preds_np=preds), f)

    print(f"Features saved to {out_path}")
