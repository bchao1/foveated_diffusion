"""Compute HPSv2 score for a directory of generated images.

Reads all metadata_*.csv files from --images_dir (same pattern as
generate_metrics.py), scores each (image, prompt) pair with HPSv2, and
writes hpsv2.json to --output_dir.

Supports multi-GPU evaluation: pass comma-separated devices such as
--device cuda:0,cuda:1 to split the workload across GPUs using
spawned subprocesses (one per GPU), which ensures CUDA_VISIBLE_DEVICES
is set before any CUDA initialisation in each worker.

Example
-------
    python eval/compute_hpsv2_score.py \\
        --images_dir /results/ours_eval/ \\
        --output_dir /tmp/hpsv2_out/ \\
        --device cuda:0,cuda:1 \\
        --batch_size 32
"""

import os
import sys
import json
import logging
import multiprocessing

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Worker (module-level so it is picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _worker(gpu_id_str, chunk_paths, chunk_prompts, batch_size, hps_version):
    """Score a chunk of images on a single GPU.

    Sets CUDA_VISIBLE_DEVICES before importing torch/hpsv2 so that the
    correct physical GPU is used.

    Parameters
    ----------
    gpu_id_str : str
        Numeric GPU index string (e.g. '0', '1').
    chunk_paths : list[str]
        Image file paths for this chunk.
    chunk_prompts : list[str]
        Corresponding prompt strings.
    batch_size : int
        Images per forward pass.
    hps_version : str
        HPSv2 model version (e.g. 'v2.1').

    Returns
    -------
    list[float]
        Per-image HPSv2 scores for this chunk.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id_str
    from metric_computation.HPSV2 import compute_hpsv2
    result = compute_hpsv2(
        chunk_paths, chunk_prompts,
        batch_size=batch_size,
        hps_version=hps_version,
        desc=f'HPSv2 GPU:{gpu_id_str}',
    )
    return result['scores']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Compute HPSv2 score for generated images and save hpsv2.json.'
    )
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory of generated images. Must contain metadata_*.csv files '
                             'with "image" and "prompt" columns.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where hpsv2.json is written.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Comma-separated PyTorch device strings '
                             '(e.g. "cuda:0" or "cuda:0,cuda:1"). Default: cuda:0.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Images per forward pass (default: 64).')
    parser.add_argument('--hps_version', type=str, default='v2.1',
                        help='HPSv2 model version (default: v2.1).')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    if not os.path.exists(args.images_dir):
        raise FileNotFoundError(f'images_dir not found: {args.images_dir}')
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load image paths and prompts from metadata_*.csv files
    # (same pattern as generate_metrics.py lines 177-198)
    # ------------------------------------------------------------------
    csv_files = sorted(
        os.path.join(args.images_dir, f)
        for f in os.listdir(args.images_dir)
        if f.startswith('metadata_') and f.endswith('.csv')
    )
    if not csv_files:
        raise FileNotFoundError(f'No metadata_*.csv files found in: {args.images_dir}')

    gen_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    image_paths = []
    prompts = []
    for _, row in gen_df.iterrows():
        gen_image_path = os.path.join(args.images_dir, row['image'])
        if not os.path.exists(gen_image_path):
            print(f'WARNING: Generated image not found: {gen_image_path}. Skipping.')
            continue
        image_paths.append(gen_image_path)
        prompts.append(row['prompt'])

    print(f'Found {len(image_paths)} images across {len(csv_files)} metadata file(s)')

    # ------------------------------------------------------------------
    # Pre-filter: skip any image that cannot be loaded (matches
    # generate_metrics.py lines 228-238)
    # ------------------------------------------------------------------
    valid_paths, valid_prompts = [], []
    for p, prompt in zip(image_paths, prompts):
        try:
            Image.open(p).load()
            valid_paths.append(p)
            valid_prompts.append(prompt)
        except Exception as e:
            print(f'WARNING: Skipping pair — cannot load image {os.path.basename(p)}: {e}')
    image_paths, prompts = valid_paths, valid_prompts

    # ------------------------------------------------------------------
    # Score: single device or multi-GPU
    # ------------------------------------------------------------------
    devices = [d.strip() for d in args.device.split(',')]

    if len(devices) == 1:
        # Single device: set CUDA_VISIBLE_DEVICES then import and run directly.
        if devices[0].startswith('cuda:'):
            os.environ['CUDA_VISIBLE_DEVICES'] = devices[0].split(':')[1]
        from metric_computation.HPSV2 import compute_hpsv2
        result = compute_hpsv2(image_paths, prompts, args.batch_size, args.hps_version)
        hpsv2_score = result['hpsv2']

    else:
        # Multiple devices: spawn one process per GPU.
        # Split (image_paths, prompts) into len(devices) roughly equal chunks.
        n = len(image_paths)
        chunk_size = (n + len(devices) - 1) // len(devices)
        chunks_paths = [image_paths[i:i + chunk_size] for i in range(0, n, chunk_size)]
        chunks_prompts = [prompts[i:i + chunk_size] for i in range(0, n, chunk_size)]

        # Each worker receives the numeric GPU ID string (e.g. '0', '1').
        worker_args = [
            (
                devices[i].split(':')[1],
                chunks_paths[i],
                chunks_prompts[i],
                args.batch_size,
                args.hps_version,
            )
            for i in range(len(chunks_paths))
        ]

        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(len(chunks_paths)) as pool:
            chunk_scores = pool.starmap(_worker, worker_args)

        # Flat mean over all per-image scores for correct weighting across
        # unequal chunks (the last chunk may be smaller).
        all_scores = [s for chunk in chunk_scores for s in chunk]
        hpsv2_score = float(np.mean(all_scores))

    # ------------------------------------------------------------------
    # Save result
    # ------------------------------------------------------------------
    out_path = os.path.join(args.output_dir, 'hpsv2.json')
    with open(out_path, 'w') as f:
        json.dump({'hpsv2': hpsv2_score}, f, indent=2)
    print(f'HPSv2: {hpsv2_score:.4f}')
    print(f'hpsv2.json saved to {out_path}')
