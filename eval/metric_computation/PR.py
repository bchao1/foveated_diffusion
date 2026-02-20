#!/usr/bin/env python3
"""PR.py -- Precision and Recall evaluation.

Standalone script; no mmgen / mmcv required.
Default model: Tero's StyleGAN TorchScript InceptionV3 (matches lakonlab).

Precision measures the fraction of generated images that fall within the
support of the real image distribution (quality).  Recall measures the
fraction of real images that fall within the support of the generated
distribution (diversity / coverage).

The implementation follows the kth-nearest-neighbour approach from
Kynkaanniemi et al. (2019, "Improved Precision and Recall Metrics for
Assessing Generative Models"), as implemented in lakonlab's InceptionMetrics
(lines 834-861) and standalone PR class (lines 338-376).

Supports loading pre-computed reference features from PKL files (including
huggingface:// URLs), matching lakonlab's reference_pkl / feats_pkl workflow.

API usage
---------
    from PR import compute_pr
    result = compute_pr(generation_dir, image_dataset_dir, args={...})
    # result = {'precision': float, 'recall': float}

Direct execution
----------------
    Edit the GEN_DIR / TRAIN_DIR / ARGS variables at the bottom of this file,
    then run:
        python PR.py
"""

import os
import sys
import random
import logging

import numpy as np
import torch

# Local utilities (same directory as this script)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inception_utils import (
    load_inception, load_reference_pkl,
    load_generation_feats_pkl, save_generation_feats_pkl,
    _list_images, extract_inception_features,
)

logger = logging.getLogger(__name__)

# Minimum number of images in each directory.
MIN_IMAGES = 10_000


# ---------------------------------------------------------------------------
# PR distance computation
# ---------------------------------------------------------------------------

def _compute_pr_distances(row_features, col_features, col_batch_size=10000):
    """Compute pairwise L2 distances between row and column feature tensors.

    Batches the column dimension to avoid OOM on large sets.
    Matches lakonlab's ``compute_pr_distances`` (line 224-232).

    Parameters
    ----------
    row_features : torch.Tensor
        Shape ``(M, D)`` -- feature vectors for the "query" set.
    col_features : torch.Tensor
        Shape ``(N, D)`` -- feature vectors for the "reference" set.
    col_batch_size : int
        Number of columns to process at a time.

    Returns
    -------
    torch.Tensor
        Shape ``(M, N)`` -- pairwise L2 distances (on CPU).
    """
    dist_batches = []
    for col_batch in col_features.split(col_batch_size):
        dist_batch = torch.cdist(
            row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        dist_batches.append(dist_batch.cpu())
    return torch.cat(dist_batches, dim=1)


def _compute_pr(real_feats_t, fake_feats_t, k=3,
                row_batch_size=10000, col_batch_size=10000):
    """Compute Precision and Recall from inception feature tensors.

    Implements the kth-nearest-neighbour approach:
      - For each point in the manifold, find the distance to its (k+1)-th
        nearest neighbour (the +1 accounts for the point itself at distance 0).
      - A probe point is "covered" if it falls within any manifold point's
        k-th nearest neighbour sphere.
      - Precision = fraction of fake points covered by the real manifold.
      - Recall    = fraction of real points covered by the fake manifold.

    Matches lakonlab's InceptionMetrics.summary() lines 834-861.

    Parameters
    ----------
    real_feats_t : torch.Tensor
        Shape ``(N_real, 2048)`` -- inception features for real images.
    fake_feats_t : torch.Tensor
        Shape ``(N_fake, 2048)`` -- inception features for generated images.
    k : int
        Number of nearest neighbours.  Default: 3.
    row_batch_size : int
        Batch size for row dimension in distance computation.
    col_batch_size : int
        Batch size for column dimension in distance computation.

    Returns
    -------
    tuple[float, float]
        ``(precision, recall)`` -- both in [0, 1].
    """
    result = {}

    for name, manifold, probes in [
        ('precision', real_feats_t, fake_feats_t),
        ('recall', fake_feats_t, real_feats_t)
    ]:
        # Step 1: For each manifold point, find distance to (k+1)-th neighbour.
        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            distance = _compute_pr_distances(
                row_features=manifold_batch,
                col_features=manifold,
                col_batch_size=col_batch_size)
            kth.append(
                distance.to(torch.float32).kthvalue(k + 1).values.to(torch.float16))
        kth = torch.cat(kth)

        # Step 2: For each probe point, check if it's within any manifold
        # point's k-th neighbour sphere.
        pred = []
        for probes_batch in probes.split(row_batch_size):
            distance = _compute_pr_distances(
                row_features=probes_batch,
                col_features=manifold,
                col_batch_size=col_batch_size)
            pred.append((distance <= kth).any(dim=1))
        result[name] = float(torch.cat(pred).to(torch.float32).mean())

    return result['precision'], result['recall']


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pr(generation_dir, image_dataset_dir, args=None):
    """Compute Precision and Recall between generated and reference images.

    Supports two modes for obtaining reference (real) features:
      1. **From images**: pass ``image_dataset_dir`` pointing to a directory of images.
      2. **From PKL**: pass ``args['reference_pkl']`` with a path or
         ``huggingface://`` URL to a pre-computed reference PKL file.
         When a reference_pkl is provided, ``image_dataset_dir`` is ignored.

    Parameters
    ----------
    generation_dir : str
        Path to the directory containing generated images.
        Must contain at least ``MIN_IMAGES`` (10 000) image files.
    image_dataset_dir : str or None
        Path to the directory containing ground-truth / reference images.
        Must contain at least ``MIN_IMAGES`` (10 000) image files.
        Ignored when ``reference_pkl`` is provided.
    args : dict, optional
        Metric-specific options.  All keys are optional:

        ``batch_size`` (int, default ``64``)
            Number of images per inception forward pass.

        ``num_images`` (int or None, default ``50000``)
            Maximum number of images to use from each directory.
            ``None`` means all available images are used.

        ``device`` (str, default ``'cuda'`` if available else ``'cpu'``)
            PyTorch device string.

        ``inception_style`` (str, default ``'StyleGAN'``)
            ``'StyleGAN'``    -- Tero's TorchScript model (default, matches lakonlab).
            ``'InceptionV3'`` -- torchvision InceptionV3.

        ``inception_path`` (str, optional)
            Only used when ``inception_style='StyleGAN'``.
            Path or URL to ``inception-2015-12-05.pt``.

        ``resize`` (bool, default ``False``)
            When True, resize images to 299x299 via torch F.interpolate bicubic
            before feeding to the inception model.

        ``reference_pkl`` (str or None, default ``None``)
            Path to a pre-computed reference PKL file containing real image
            features. Supports local paths, ``huggingface://`` URLs, and
            ``http(s)://`` URLs.  When provided, ``image_dataset_dir`` is ignored.
            PKL must contain ``'real_feats_np'`` key with shape ``(N, 2048)``.

        ``k`` (int, default ``3``)
            Number of nearest neighbours for the kNN manifold estimation.

        ``row_batch_size`` (int, default ``10000``)
            Batch size for row dimension in pairwise distance computation.

        ``col_batch_size`` (int, default ``10000``)
            Batch size for column dimension in pairwise distance computation.

        ``equal`` (bool, default ``False``)
            When ``True`` and the training directory has *more* images than
            the generation directory, randomly sub-sample the training images
            so that both sets have the same size.

        ``seed`` (int, default ``42``)
            Random seed used for the ``equal`` sub-sampling.

        ``generation_feats_pkl`` (str or None, default ``None``)
            If set, path to a PKL file to cache inception features for the
            generation directory.  When the file exists, features are loaded
            from it instead of re-extracting; when the file does not exist,
            features are extracted and then saved to this path for future runs.

    Returns
    -------
    dict
        ``'precision'`` (float) -- fraction of generated images within the
                        real manifold support (higher is better).
        ``'recall'``    (float) -- fraction of real images within the
                        generated manifold support (higher is better).

    Raises
    ------
    ValueError
        If either directory contains fewer than ``MIN_IMAGES`` images.
    """
    if args is None:
        args = {}

    batch_size      = int(args.get('batch_size', 64))
    num_images      = args.get('num_images', 50000)
    device          = args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    inception_style = args.get('inception_style', 'StyleGAN')
    inception_path  = args.get('inception_path', None)
    resize          = bool(args.get('resize', False))
    reference_pkl   = args.get('reference_pkl', None)
    k               = int(args.get('k', 3))
    row_batch_size  = int(args.get('row_batch_size', 10000))
    col_batch_size  = int(args.get('col_batch_size', 10000))
    equal           = bool(args.get('equal', False))
    seed            = int(args.get('seed', 42))
    generation_feats_pkl = args.get('generation_feats_pkl', None)

    # -- Load inception model --------------------------------------------------
    logger.info(f'Loading Inception model (style={inception_style}) ...')
    inception_args = dict(type=inception_style)
    if inception_path is not None:
        inception_args['inception_path'] = inception_path
    inception_net, inception_style = load_inception(
        inception_args, metric='FID', map_location='cpu'
    )
    inception_net = inception_net.to(device).eval()

    # -- Load reference PKL or collect real image paths ------------------------
    if reference_pkl is not None:
        logger.info(f'Using reference PKL: {reference_pkl}')
        ref = load_reference_pkl(reference_pkl)
        real_feats_np = ref['real_feats_np']
        logger.info(f'Reference features loaded: {real_feats_np.shape}')
    else:
        real_feats_np = None

    # -- Collect image paths ---------------------------------------------------
    gen_paths = _list_images(generation_dir)

    if len(gen_paths) == 0:
        raise ValueError(f'No images found in generation_dir: {generation_dir!r}')
    if len(gen_paths) < MIN_IMAGES:
        raise ValueError(
            f'generation_dir must contain at least {MIN_IMAGES} images, '
            f'found {len(gen_paths)}: {generation_dir!r}'
        )

    if reference_pkl is None:
        real_paths = _list_images(image_dataset_dir)
        if len(real_paths) == 0:
            raise ValueError(f'No images found in image_dataset_dir: {image_dataset_dir!r}')
        if len(real_paths) < MIN_IMAGES:
            raise ValueError(
                f'image_dataset_dir must contain at least {MIN_IMAGES} images, '
                f'found {len(real_paths)}: {image_dataset_dir!r}'
            )

    # -- Equal subsampling -----------------------------------------------------
    if reference_pkl is None and equal and len(real_paths) > len(gen_paths):
        rng = random.Random(seed)
        real_paths = rng.sample(real_paths, len(gen_paths))
        logger.info(
            f'equal=True: sub-sampled training images to {len(real_paths)} '
            f'(seed={seed})'
        )

    # -- Optional hard cap on image count --------------------------------------
    if num_images is not None:
        gen_paths = gen_paths[:num_images]
        if reference_pkl is None:
            real_paths = real_paths[:num_images]
        else:
            real_feats_np = real_feats_np[:num_images]

    logger.info(f'Generated images  : {len(gen_paths)}')
    if reference_pkl is None:
        logger.info(f'Reference images  : {len(real_paths)}')
    else:
        logger.info(f'Reference features: {real_feats_np.shape[0]}')

    # -- Extract or load generation inception features --------------------------
    if generation_feats_pkl and os.path.isfile(generation_feats_pkl):
        cached = load_generation_feats_pkl(generation_feats_pkl, num_images=num_images)
        gen_feats = cached['feats_np']
        logger.info(f'Using cached generation features: {gen_feats.shape[0]} samples')
    else:
        logger.info('Extracting features from generated images ...')
        gen_feats, gen_preds = extract_inception_features(
            gen_paths, inception_net, inception_style, batch_size, device, resize=resize
        )
        if generation_feats_pkl:
            save_generation_feats_pkl(generation_feats_pkl, gen_feats, gen_preds)

    if reference_pkl is None:
        logger.info('Extracting features from reference images ...')
        real_feats_np, _ = extract_inception_features(
            real_paths, inception_net, inception_style, batch_size, device, resize=resize
        )

    # -- Compute Precision and Recall ------------------------------------------
    logger.info('Computing Precision and Recall ...')
    real_feats_t = torch.from_numpy(real_feats_np)
    fake_feats_t = torch.from_numpy(gen_feats)

    precision, recall = _compute_pr(
        real_feats_t, fake_feats_t,
        k=k, row_batch_size=row_batch_size, col_batch_size=col_batch_size
    )

    return dict(precision=precision, recall=recall)


# ---------------------------------------------------------------------------
# Direct-execution entry point
# ---------------------------------------------------------------------------

def _print_results(results):
    width = 52
    print()
    print('=' * width)
    print(f"{'PR Results':^{width}}")
    print('=' * width)
    print(f"  {'Precision (higher is better)':<30} {results['precision']:>10.5f}")
    print(f"  {'Recall    (higher is better)':<30} {results['recall']:>10.5f}")
    print('=' * width)
    print()


if __name__ == '__main__':
    # -- Configure these variables before running ------------------------------
    GEN_DIR           = '/path/to/generated/images'    # directory of generated images
    IMAGE_DATASET_DIR = '/path/to/reference/images'    # directory of GT/real images

    ARGS = {
        # Number of images per inception forward pass.
        'batch_size': 64,

        # Hard cap on images used from each directory.
        # Standard practice: 50000.  Set to None to use all available images.
        'num_images': 50000,

        # Compute device.  Set to None for automatic selection
        # ('cuda' if available, else 'cpu').
        'device': None,

        # Inception model type.
        # 'StyleGAN' (default) -- Tero's TorchScript model (matches lakonlab).
        # 'InceptionV3'        -- torchvision InceptionV3.
        'inception_style': 'StyleGAN',

        # Path or URL to Tero's inception-2015-12-05.pt (only for StyleGAN style).
        # Set to None to auto-download from NVIDIA CDN.
        'inception_path': None,

        # Whether to resize images to 299x299 before feeding to inception.
        # False (default) -- images are assumed to be at correct resolution.
        # True -- resize via torch F.interpolate bicubic with antialias.
        'resize': False,

        # Pre-computed reference PKL file (optional).
        # When provided, training_dir is ignored and features are loaded from PKL.
        # Supports local paths, huggingface:// URLs, and http(s):// URLs.
        'reference_pkl': None,

        # Number of nearest neighbours for kNN manifold estimation.
        'k': 3,

        # Batch sizes for pairwise distance computation.
        'row_batch_size': 10000,
        'col_batch_size': 10000,

        # When True, randomly sub-sample training images to match the count
        # of generated images (only when training_dir has MORE images).
        'equal': False,

        # Random seed used for 'equal' sub-sampling.
        'seed': 42,

        # Cache generation inception features to this PKL path (load if exists, else save after extraction).
        'generation_feats_pkl': None,
    }
    # --------------------------------------------------------------------------

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%H:%M:%S',
    )

    results = compute_pr(GEN_DIR, IMAGE_DATASET_DIR, ARGS)
    _print_results(results)
