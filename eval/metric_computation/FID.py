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
import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

# Local utilities (same directory as this script)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inception_utils import (
    load_inception, load_reference_pkl,
    load_generation_feats_pkl, save_generation_feats_pkl,
    _list_images, extract_inception_features,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def _calc_fid(fake_mean, fake_cov, real_mean, real_cov, eps=1e-6):
    """Compute FID between two multivariate Gaussians.

    Implements the standard FID formula::

        FID = ||mu_fake - mu_real||^2
              + Tr(Sigma_fake + Sigma_real - 2*sqrt(Sigma_fake @ Sigma_real))
    
    Matches https://github.com/Lakonik/piFlow/blob/67431c35d88779088acec23b38773402ace0fd94/lakonlab/evaluation/metrics.py#L642

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
        print('product of cov matrices is singular')
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

def compute_fid(gen_paths, real_paths, args=None,
                foveation_mask=None, foveate_real=False, downsample_real=False,
                device=None, real_pil_transform=None, real_image_transform=None):
    """Compute FID between generated images and reference images.

    Supports two modes for obtaining reference (real) statistics:
      1. **From images**: pass ``real_paths`` as a list of image file paths.
      2. **From PKL**: pass ``args['reference_pkl']`` with a path or
         ``huggingface://`` URL to a pre-computed reference PKL file.
         When a reference_pkl is provided, ``real_paths`` is ignored.

    Parameters
    ----------
    gen_paths : list[str]
        Ordered list of generated image file paths.
    real_paths : list[str] or None
        Ordered list of ground-truth / reference image file paths.
        Ignored when ``reference_pkl`` is provided.
    args : dict, optional
        Metric-specific options.  All keys are optional:

        ``batch_size`` (int, default ``64``)
            Number of images per inception forward pass.

        ``num_images`` (int or None, default ``None``)
            Maximum number of images to use.
            ``None`` means all available images are used.

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
            URLs, and ``http(s)://`` URLs. When provided, ``real_paths``
            is ignored. Default is
            ``'huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl'``
            so standard FID evaluation can run with only ``gen_paths``
            (same as lakonlab InceptionMetrics). Set to ``None`` to use
            ``real_paths`` instead.

        ``generation_feats_pkl`` (str or None, default ``None``)
            If set, path to a PKL file to cache inception features for the
            generated images.  When the file exists, features are loaded
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

    foveation_mask : np.ndarray or None, optional
        Binary mask of shape ``(H, W)`` (same resolution as the images), where
        1 = high-resolution (foveal) region and 0 = peripheral region.
        Applied to real images when ``foveate_real=True``.  Default: None.
    foveate_real : bool, optional
        When True and ``foveation_mask`` is not None, apply foveated blending to
        real images: the foveal region (mask=1) is kept at full resolution while
        the peripheral region (mask=0) is replaced with a bicubic
        downsample-then-upsample of the full image.  Default: False.
    downsample_real : bool, optional
        When True, bicubic 2× downsample each real image from ``(H, W)`` to
        ``(H/2, W/2)`` before feature extraction.  Overrides
        ``foveation_mask`` / ``foveate_real``.  Default: False.
    device : str or None, optional
        PyTorch device string (e.g. ``'cuda:0'``, ``'cpu'``).  When provided,
        takes precedence over ``args['device']``.  Default: None (auto-select).
    """
    if args is None:
        args = {}

    batch_size      = int(args.get('batch_size', 64))
    num_images      = args.get('num_images', None)
    if device is None:
        device      = args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    inception_style = args.get('inception_style', 'StyleGAN')
    inception_path  = args.get('inception_path', None)
    resize          = bool(args.get('resize', False))
    _default_ref_pkl = 'huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl'
    reference_pkl   = args.get('reference_pkl', _default_ref_pkl)
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

    # -- Load reference PKL or validate real image paths -----------------------
    if reference_pkl is not None:
        logger.info(f'Using reference PKL: {reference_pkl}')
        ref = load_reference_pkl(reference_pkl)
        real_mean = ref['mean']
        real_cov = ref['cov']
        logger.info(f'Reference stats loaded (num_real_feeded={ref.get("num_real_feeded", "?")})')
    else:
        real_mean = None
        real_cov = None
        if not real_paths:
            raise ValueError('real_paths is empty and no reference_pkl was provided.')

    # -- Optional hard cap on image count --------------------------------------
    if num_images is not None:
        gen_paths = gen_paths[:num_images]

    if len(gen_paths) == 0:
        raise ValueError('gen_paths is empty.')

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

    # Build real-image transform from foveation flags.
    # real_image_transform (pre-built per-image transform) takes priority when provided.
    real_transform = None
    if real_image_transform is not None:
        real_transform = real_image_transform
    elif downsample_real:
        def real_transform(batch):
            H, W = batch.shape[2], batch.shape[3]
            return F.interpolate(
                batch, size=(H // 2, W // 2),
                mode='bicubic', align_corners=False, antialias=True,
            ).clamp(-1.0, 1.0)
    elif foveation_mask is not None and foveate_real:
        _mask = torch.as_tensor(foveation_mask, dtype=torch.float32).to(device)
        if _mask.dim() == 2:
            _mask = _mask[None, None]  # (1, 1, H_orig, W_orig)
        def real_transform(batch):
            H, W = batch.shape[2], batch.shape[3]
            # Resize mask to match actual batch spatial dims (handles resize=True at 299x299).
            mask_hw = F.interpolate(_mask, size=(H, W), mode='bilinear', align_corners=False)
            low_res = F.interpolate(
                batch, size=(H // 2, W // 2),
                mode='bicubic', align_corners=False, antialias=True,
            ).clamp(-1.0, 1.0)
            low_res_up = F.interpolate(
                low_res, size=(H, W),
                mode='bicubic', align_corners=False, antialias=True,
            ).clamp(-1.0, 1.0)
            return low_res_up * (1.0 - mask_hw) + batch * mask_hw

    if reference_pkl is None:
        logger.info('Extracting features from reference images ...')
        real_feats, _ = extract_inception_features(
            real_paths, inception_net, inception_style, batch_size, device,
            resize=resize, image_transform=real_transform,
            pil_transform=real_pil_transform,
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

        # Hard cap on images used.
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
        # When provided, real_paths is ignored. Set to None to use real_paths instead.
        'reference_pkl': 'huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl',

        # Cache generation inception features to this PKL path (load if exists, else save after extraction).
        'generation_feats_pkl': None,
    }
    # --------------------------------------------------------------------------

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%H:%M:%S',
    )

    gen_paths  = _list_images(GEN_DIR)
    real_paths = _list_images(IMAGE_DATASET_DIR)
    results = compute_fid(gen_paths, real_paths, ARGS)
    _print_results(results)
