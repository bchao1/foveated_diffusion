#!/usr/bin/env python3
"""FID.py -- Frechet Inception Distance evaluation.

Standalone script; no mmgen / mmcv required.
Default model: Tero's StyleGAN TorchScript InceptionV3 (matches lakonlab).
torchvision InceptionV3 is also supported via inception_style='InceptionV3'.

Supports loading pre-computed reference statistics from PKL files (including
huggingface:// URLs), matching lakonlab's reference_pkl workflow.

API usage
---------
    from FID import compute_fid
    result = compute_fid(generation_dir, image_dataset_dir, args={...})
    # result = {'fid': float, 'mean_dist': float, 'cov_dist': float}

Direct execution
----------------
    Edit the GEN_DIR / TRAIN_DIR / ARGS variables at the bottom of this file,
    then run:
        python FID.py
"""

import os
import sys
import random
import logging

import numpy as np
import torch
from scipy import linalg

# Local utilities (same directory as this script)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inception_utils import (
    load_inception, load_reference_pkl,
    load_generation_feats_pkl, save_generation_feats_pkl,
    _list_images, extract_inception_features,
)

logger = logging.getLogger(__name__)

# Minimum number of images that must be present in each directory. 10,000 is the standard minimum.
# The common practice is to use 50,000 images, but the MIN_IMAGES variable here is just for assertion of bare minimum.
MIN_IMAGES = 10_000


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def _calc_fid(fake_mean, fake_cov, real_mean, real_cov, eps=1e-6):
    """Compute FID between two multivariate Gaussians.

    Implements the standard FID formula::

        FID = ||mu_fake - mu_real||^2
              + Tr(Sigma_fake + Sigma_real - 2*sqrt(Sigma_fake @ Sigma_real))

    Parameters
    ----------
    fake_mean : np.ndarray  shape (2048,)
    fake_cov  : np.ndarray  shape (2048, 2048)
    real_mean : np.ndarray  shape (2048,)
    real_cov  : np.ndarray  shape (2048, 2048)
    eps : float
        Small offset added to the diagonal when the product covariance is
        singular (prevents non-finite sqrtm output).

    Returns
    -------
    tuple[float, float, float]
        ``(fid, mean_dist, cov_dist)`` where ``fid = mean_dist + cov_dist``.

        ``mean_dist`` -- squared L2 distance between the two mean vectors:
            ``||mu_gen - mu_real||^2``
        ``cov_dist``  -- trace-based distance between the two covariance matrices:
            ``Tr(Sigma_gen) + Tr(Sigma_real) - 2*Tr(sqrt(Sigma_gen @ Sigma_real))``

        Neither value is the mean or covariance of either image set on its own;
        both are *distance* measures that jointly involve generated and real stats.
    """
    cov_sqrt, _ = linalg.sqrtm(fake_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        offset = np.eye(fake_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((fake_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            raise ValueError(
                f'Imaginary component too large: {np.max(np.abs(cov_sqrt.imag)):.4f}'
            )
        cov_sqrt = cov_sqrt.real

    mean_diff = fake_mean - real_mean
    mean_dist = float(mean_diff @ mean_diff)
    cov_dist  = float(
        np.trace(fake_cov) + np.trace(real_cov) - 2.0 * np.trace(cov_sqrt)
    )
    fid = mean_dist + cov_dist
    return fid, mean_dist, cov_dist


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_fid(generation_dir, image_dataset_dir, args=None):
    """Compute FID between generated images and reference images.

    Supports two modes for obtaining reference (real) statistics:
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
            Applied *after* the 10 K minimum check and *after* ``equal``
            subsampling.

        ``device`` (str, default ``'cuda'`` if available else ``'cpu'``)
            PyTorch device string, e.g. ``'cuda'``, ``'cuda:1'``, ``'cpu'``.

        ``inception_style`` (str, default ``'StyleGAN'``)
            ``'StyleGAN'``    -- Tero's TorchScript model (default, matches lakonlab).
            ``'InceptionV3'`` -- torchvision InceptionV3.

        ``inception_path`` (str, optional)
            Only used when ``inception_style='StyleGAN'``.
            Path or URL to ``inception-2015-12-05.pt``.

        ``resize`` (bool, default ``False``)
            When True, resize images to 299x299 via torch F.interpolate bicubic
            before feeding to the inception model. When False, images are assumed
            to already be at the correct resolution.

        ``reference_pkl`` (str or None, default see below)
            Path to a pre-computed reference PKL file containing real image
            statistics (mean, cov). Supports local paths, ``huggingface://``
            URLs, and ``http(s)://`` URLs. When provided, ``image_dataset_dir``
            is ignored. Default is
            ``'huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl'``
            so standard FID evaluation can run with only ``generation_dir``
            (same as lakonlab InceptionMetrics). Set to ``None`` to use
            ``training_dir`` instead.

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
        ``'fid'``       (float) -- Frechet Inception Distance (lower is better).
        ``'mean_dist'`` (float) -- squared L2 distance between the mean inception
                        feature vectors of generated and real images.
        ``'cov_dist'``  (float) -- trace-based distance between the covariance
                        matrices of generated and real features.
        Note: ``fid = mean_dist + cov_dist``.

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
    _default_ref_pkl = 'huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl'
    reference_pkl   = args.get('reference_pkl', _default_ref_pkl)
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
        real_mean = ref['mean']
        real_cov = ref['cov']
        logger.info(f'Reference stats loaded (num_real_feeded={ref.get("num_real_feeded", "?")})')
    else:
        real_mean = None
        real_cov = None

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

    logger.info(f'Generated images  : {len(gen_paths)}')
    if reference_pkl is None:
        logger.info(f'Reference images  : {len(real_paths)}')

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
        real_feats, _ = extract_inception_features(
            real_paths, inception_net, inception_style, batch_size, device, resize=resize
        )
        real_mean = np.mean(real_feats, axis=0)
        real_cov = np.cov(real_feats, rowvar=False)

    # -- Compute Gaussian statistics and FID -----------------------------------
    gen_mean = np.mean(gen_feats, axis=0)
    gen_cov  = np.cov(gen_feats, rowvar=False)

    logger.info('Computing FID ...')
    fid, mean_dist, cov_dist = _calc_fid(gen_mean, gen_cov, real_mean, real_cov)

    return dict(fid=fid, mean_dist=mean_dist, cov_dist=cov_dist)


# ---------------------------------------------------------------------------
# Direct-execution entry point
# ---------------------------------------------------------------------------

def _print_results(results):
    width = 52
    print()
    print('=' * width)
    print(f"{'FID Results':^{width}}")
    print('=' * width)
    print(f"  {'FID (lower is better)':<30} {results['fid']:>10.4f}")
    print(f"  {'Mean dist  ||mu_gen-mu_real||2':<30} {results['mean_dist']:>10.4f}")
    print(f"  {'Cov dist   (trace term)':<30} {results['cov_dist']:>10.4f}")
    print('=' * width)
    print()


if __name__ == '__main__':
    # -- Configure these variables before running ------------------------------
    GEN_DIR            = '/path/to/generated/images'    # directory of generated images
    IMAGE_DATASET_DIR  = '/path/to/reference/images'    # directory of GT/real images

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

        # Pre-computed reference PKL file. Default uses ImageNet-256 reference (same as lakonlab).
        # When provided, training_dir is ignored. Set to None to use training_dir instead.
        'reference_pkl': 'huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl',

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

    results = compute_fid(GEN_DIR, IMAGE_DATASET_DIR, ARGS)
    _print_results(results)
