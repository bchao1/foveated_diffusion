#!/usr/bin/env python3
"""CLIP.py -- CLIP Score (image-text similarity) evaluation.

Standalone script; no mmgen / mmcv required.
Default model: ViT-L-14-336-quickgelu (OpenAI pretrained) via open_clip.

CLIP Score measures prompt alignment by computing the average cosine
similarity between CLIP image embeddings and CLIP text embeddings.
Higher is better (range [-1, 1], typically 0.2-0.4 for good models).

This metric does NOT use training_dir; that argument is accepted only
for API consistency with the other metrics in this folder.

Prompts can be provided in three ways (checked in this order):
  1. ``args['prompts']`` -- a list of prompt strings, one per image
     (in the same sorted order as image files in generation_dir).
  2. ``args['prompts_file']`` -- path to a text file with one prompt per line,
     corresponding to sorted image filenames.
  3. A ``prompts.txt`` file in ``generation_dir`` itself.

API usage
---------
    from CLIP import compute_clip_score
    result = compute_clip_score(generation_dir, image_dataset_dir=None, args={...})
    # result = {'clipsim': float}

Direct execution
----------------
    Edit the GEN_DIR / ARGS variables at the bottom of this file, then run:
        python CLIP.py
"""

import os
import sys
import logging
import contextlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# Minimum number of images.
MIN_IMAGES = 10_000


# ---------------------------------------------------------------------------
# OpenCLIP loading with logging quarantine
# ---------------------------------------------------------------------------

# Module-level cache to avoid reloading the model.
_clip_cache = {}


@contextlib.contextmanager
def _quarantine_openclip_logging():
    """Guard against open_clip mutating global logging.

    Snapshots root handlers/level, runs the block, then removes any
    NEW handlers and restores the level. Also disables propagation
    for the open_clip logger so logs don't bubble to root.
    Matches lakonlab's _quarantine_openclip_logging (lines 55-83).
    """
    root = logging.getLogger()
    before_ids = {id(h) for h in root.handlers}
    before_level = root.level

    try:
        yield
    finally:
        for h in list(root.handlers):
            if id(h) not in before_ids:
                root.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
        root.setLevel(before_level)

        oc = logging.getLogger("open_clip")
        oc.propagate = False
        oc.handlers.clear()


def _load_openclip(model_name='ViT-L-14-336-quickgelu',
                   pretrained='openai',
                   device='cpu',
                   precision='fp16'):
    """Load an OpenCLIP model and tokenizer with caching.

    Matches lakonlab's load_openclip (lines 202-221).

    Parameters
    ----------
    model_name : str
        OpenCLIP model architecture name.
    pretrained : str
        Pretrained weights tag (e.g. ``'openai'``, ``'laion2b_s39b_b160k'``).
    device : str
        Torch device string.
    precision : str
        ``'fp16'``, ``'bf16'``, or ``'fp32'``.

    Returns
    -------
    tuple
        ``(model, tokenizer)``
    """
    cache_key = f'{model_name}_{pretrained}_{device}_{precision}'
    if cache_key in _clip_cache:
        return _clip_cache[cache_key]

    from open_clip import create_model, get_tokenizer

    with _quarantine_openclip_logging():
        model = create_model(
            model_name,
            pretrained=pretrained,
            precision=precision,
            device=device,
            output_dict=True)
        model.requires_grad_(False)
        tokenizer = get_tokenizer(model_name)

    _clip_cache[cache_key] = (model, tokenizer)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Image loading and preprocessing
# ---------------------------------------------------------------------------

# Supported image file extensions.
_IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def _list_images(directory):
    """Return a sorted list of all image file paths in *directory*."""
    paths = []
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname)[1].lower() in _IMG_EXTS:
            paths.append(os.path.join(directory, fname))
    return paths


def _load_image_01(path):
    """Load a single image as float32 tensor in [0, 1].

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    torch.Tensor
        Shape ``(3, H, W)``, dtype ``float32``, values in ``[0, 1]``.
    """
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _resize_min_side_then_center_crop(imgs, image_size, image_mean, image_std, dtype):
    """Resize so min(H,W) == image_size, center crop, normalize, cast dtype.

    Matches lakonlab's CLIPSimilarity._resize_min_side_then_center_crop
    (lines 1200-1231).

    Parameters
    ----------
    imgs : torch.Tensor
        Shape ``(B, 3, H, W)`` in ``[0, 1]``.
    image_size : int
        Target crop size (e.g. 336 for ViT-L/14@336).
    image_mean : torch.Tensor
        Shape ``(3, 1, 1)`` -- model's per-channel mean.
    image_std : torch.Tensor
        Shape ``(3, 1, 1)`` -- model's per-channel std.
    dtype : torch.dtype
        Target dtype (e.g. ``torch.float16``).

    Returns
    -------
    torch.Tensor
        Shape ``(B, 3, image_size, image_size)``, normalized and cast.
    """
    _, _, H, W = imgs.shape
    target = image_size

    short = min(H, W)
    if short == 0:
        raise ValueError("Invalid image with zero dimension.")
    scale = target / float(short)

    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))
    if new_h != H or new_w != W:
        imgs = F.interpolate(
            imgs, size=(new_h, new_w),
            mode='bicubic', align_corners=False, antialias=True
        ).clamp(0, 1)

    top = max(0, (new_h - target) // 2)
    left = max(0, (new_w - target) // 2)
    imgs = imgs[:, :, top:top + target, left:left + target]

    imgs = (imgs - image_mean) / image_std
    return imgs.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_prompts(generation_dir, args):
    """Load prompts from args or from a file in the generation directory.

    Checked in order:
      1. ``args['prompts']`` -- a list of strings.
      2. ``args['prompts_file']`` -- path to a .txt file (one prompt per line).
      3. ``generation_dir/prompts.txt`` -- fallback auto-discovery.

    Parameters
    ----------
    generation_dir : str
    args : dict

    Returns
    -------
    list[str]
        List of prompt strings.

    Raises
    ------
    ValueError
        If no prompts can be found.
    """
    # Option 1: prompts passed directly
    if 'prompts' in args and args['prompts'] is not None:
        return list(args['prompts'])

    # Option 2: explicit prompts file path
    prompts_file = args.get('prompts_file', None)
    if prompts_file is None:
        # Option 3: auto-discover prompts.txt in generation_dir
        candidate = os.path.join(generation_dir, 'prompts.txt')
        if os.path.isfile(candidate):
            prompts_file = candidate

    if prompts_file is not None:
        logger.info(f'Loading prompts from: {prompts_file}')
        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts

    raise ValueError(
        'No prompts found. Provide one of:\n'
        "  1. args['prompts'] -- list of strings\n"
        "  2. args['prompts_file'] -- path to a .txt file (one prompt per line)\n"
        f"  3. A 'prompts.txt' file in generation_dir ({generation_dir})"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_clip_score(generation_dir, image_dataset_dir, args=None):
    """Compute average CLIP cosine similarity between images and prompts.

    This metric measures prompt alignment: how well the generated images
    match the text prompts used to generate them.  It does **not** use
    reference images; ``image_dataset_dir`` is accepted only for API consistency.

    Parameters
    ----------
    generation_dir : str
        Path to the directory containing generated images.
        Must contain at least ``MIN_IMAGES`` (10 000) image files.
    image_dataset_dir : str or None
        Unused.  Accepted for API consistency.
    args : dict, optional
        Metric-specific options.  All keys are optional:

        ``batch_size`` (int, default ``64``)
            Number of images per forward pass.

        ``num_images`` (int or None, default ``50000``)
            Maximum number of images to evaluate.  ``None`` uses all.

        ``device`` (str, default ``'cuda'`` if available else ``'cpu'``)
            PyTorch device string.

        ``model_name`` (str, default ``'ViT-L-14-336-quickgelu'``)
            OpenCLIP model architecture.

        ``pretrained`` (str, default ``'openai'``)
            Pretrained weights tag.

        ``precision`` (str, default ``'fp16'``)
            Model precision: ``'fp16'``, ``'bf16'``, or ``'fp32'``.

        ``prompts`` (list[str] or None, default ``None``)
            List of prompt strings, one per image (sorted order).

        ``prompts_file`` (str or None, default ``None``)
            Path to a text file with one prompt per line.
            Falls back to ``generation_dir/prompts.txt``.

    Returns
    -------
    dict
        ``'clipsim'`` (float) -- mean cosine similarity in [-1, 1].
            Typically 0.2-0.4 for good text-to-image models.

    Raises
    ------
    ValueError
        If no prompts can be found, or if prompt count != image count,
        or if generation_dir contains fewer than ``MIN_IMAGES`` images.
    """
    if args is None:
        args = {}

    batch_size  = int(args.get('batch_size', 64))
    num_images  = args.get('num_images', 50000)
    device      = args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model_name  = args.get('model_name', 'ViT-L-14-336-quickgelu')
    pretrained  = args.get('pretrained', 'openai')
    precision   = args.get('precision', 'fp16')

    dtype = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32
    }.get(precision, torch.float16)

    # -- Load CLIP model -------------------------------------------------------
    logger.info(f'Loading CLIP model ({model_name}, {pretrained}) ...')
    model, tokenizer = _load_openclip(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        precision=precision,
    )
    model.eval()

    # Get model's image size and normalization stats
    image_size = model.visual.image_size
    if isinstance(image_size, tuple):
        assert len(image_size) == 2 and image_size[0] == image_size[1]
        image_size = image_size[0]
    image_size = int(image_size)

    image_mean = torch.tensor(model.visual.image_mean, device=device).view(3, 1, 1)
    image_std = torch.tensor(model.visual.image_std, device=device).view(3, 1, 1)

    # -- Collect image paths ---------------------------------------------------
    gen_paths = _list_images(generation_dir)

    if len(gen_paths) == 0:
        raise ValueError(f'No images found in generation_dir: {generation_dir!r}')
    if len(gen_paths) < MIN_IMAGES:
        raise ValueError(
            f'generation_dir must contain at least {MIN_IMAGES} images, '
            f'found {len(gen_paths)}: {generation_dir!r}'
        )

    # -- Load prompts ----------------------------------------------------------
    prompts = _load_prompts(generation_dir, args)

    # -- Optional hard cap on image count --------------------------------------
    if num_images is not None:
        gen_paths = gen_paths[:num_images]
        prompts = prompts[:num_images]

    if len(prompts) != len(gen_paths):
        raise ValueError(
            f'Number of prompts ({len(prompts)}) does not match number of '
            f'images ({len(gen_paths)}). Ensure one prompt per image.'
        )

    logger.info(f'Generated images : {len(gen_paths)}')
    logger.info(f'Prompts          : {len(prompts)}')

    # -- Compute CLIP similarities in batches ----------------------------------
    all_sims = []
    n = len(gen_paths)

    for start in range(0, n, batch_size):
        batch_paths = gen_paths[start:start + batch_size]
        batch_prompts = prompts[start:start + batch_size]

        # Load images as [0, 1] float32
        imgs = [_load_image_01(p) for p in batch_paths]
        imgs = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)

        # Preprocess: resize min side, center crop, normalize
        imgs = _resize_min_side_then_center_crop(
            imgs, image_size, image_mean, image_std, dtype
        )

        # Tokenize prompts
        text = tokenizer(batch_prompts).to(device=device)

        with torch.no_grad():
            # Forward pass -- model was created with output_dict=True
            out = model(imgs, text)
            if isinstance(out, dict) and 'image_features' in out and 'text_features' in out:
                img_feat = out['image_features']
                txt_feat = out['text_features']
            else:
                img_feat = model.encode_image(imgs)
                txt_feat = model.encode_text(text)

            # Cosine similarity per pair
            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)
            sim = (img_feat * txt_feat).sum(dim=-1).to(torch.float32)  # (B,)

        all_sims.append(sim.cpu())

        end = min(start + batch_size, n)
        if end % (batch_size * 10) == 0 or end == n:
            logger.info(f'  {end}/{n} image-text pairs processed')

    # -- Aggregate results -----------------------------------------------------
    sims = torch.cat(all_sims, dim=0)
    mean_sim = float(sims.mean())

    return dict(clipsim=mean_sim)


# ---------------------------------------------------------------------------
# Direct-execution entry point
# ---------------------------------------------------------------------------

def _print_results(results):
    width = 52
    print()
    print('=' * width)
    print(f"{'CLIP Score Results':^{width}}")
    print('=' * width)
    print(f"  {'CLIP Sim (higher is better)':<30} {results['clipsim']:>10.4f}")
    print('=' * width)
    print()


if __name__ == '__main__':
    # -- Configure these variables before running ------------------------------
    GEN_DIR = '/path/to/generated/images'    # directory of generated images
    # image_dataset_dir is unused by CLIP Score; set to None
    IMAGE_DATASET_DIR = None

    ARGS = {
        # Number of images per forward pass.
        'batch_size': 64,

        # Hard cap on images evaluated.
        # Standard practice: 50000.  Set to None to use all.
        'num_images': 50000,

        # Compute device.  Set to None for automatic selection.
        'device': None,

        # OpenCLIP model name and pretrained weights.
        'model_name': 'ViT-L-14-336-quickgelu',
        'pretrained': 'openai',

        # Model precision: 'fp16', 'bf16', or 'fp32'.
        'precision': 'fp16',

        # Path to a text file with one prompt per line (sorted order matches images).
        # Set to None to auto-discover generation_dir/prompts.txt.
        'prompts_file': None,
    }
    # --------------------------------------------------------------------------

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%H:%M:%S',
    )

    results = compute_clip_score(GEN_DIR, IMAGE_DATASET_DIR, ARGS)
    _print_results(results)
