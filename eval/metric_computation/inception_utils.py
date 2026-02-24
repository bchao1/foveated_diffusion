# inception_utils.py
# Inception model loading and feature extraction utilities for FID / IS / PR computation.
# Adapted from lakonlab/evaluation/metrics.py -- all mmgen/mmcv dependencies removed.
#
# -- Inception model variants --------------------------------------------------
#
# Two inception model types are supported:
#
#   1. Tero's StyleGAN TorchScript model  (type='StyleGAN', DEFAULT)
#      -----------------------------------------------------------------
#      C++ / TorchScript InceptionV3 by Tero Karras (NVlabs), used in
#      official StyleGAN FID evaluations and by lakonlab's InceptionMetrics.
#      This is the default to match lakonlab's evaluation pipeline.
#      Input: uint8 in [0, 255], shape (B, 3, H, W).
#      Features: 2048-d pool_3.  Class probs: 1008-d softmax.
#
#   2. torchvision InceptionV3  (type='InceptionV3')
#      -----------------------------------------------
#      Standard PyTorch InceptionV3 with ImageNet pretrained weights.
#      Input: float32 in [0, 1], shape (B, 3, 299, 299).
#      The model's internal transform_input normalises to the expected range.
#      Features: 2048-d pool layer.  Class probs: 1000-d softmax.
#      NOTE: FID scores from this model are NOT comparable to StyleGAN.
#
# -- Preprocessing -------------------------------------------------------------
#
#   Images are loaded via PIL, converted to RGB, and normalised to [-1, 1].
#   When resize=True, torch F.interpolate bicubic with antialias is applied
#   to resize to 299x299, matching lakonlab's extract_features (line 708-710).
#   When resize=False (our standalone default), images are assumed to already
#   be at the desired resolution -- no resize is performed.
#   For the StyleGAN model, the [-1, 1] tensor is converted to [0, 255] uint8
#   via (x * 127.5 + 128).clamp(0,255).to(uint8), matching lakonlab exactly.
#
# -- Reference PKL files -------------------------------------------------------
#
#   Pre-computed inception features can be loaded from PKL files via
#   load_reference_pkl(). Supports local paths and huggingface:// URLs.
#   PKL format: {'mean': np.ndarray, 'cov': np.ndarray,
#                'real_feats_np': np.ndarray, 'num_real_feeded': int}
#
# -- Shared helpers ------------------------------------------------------------
#
#   _IMG_EXTS                     -- set of supported image file extensions
#   _list_images(dir)             -- sorted list of image paths in a directory
#   _load_image_tensor(p)         -- PIL load -> RGB -> [-1, 1] float32 (no resize)
#   extract_inception_features()  -- batch feature extraction -> (feat_np, pred_np)
#   load_reference_pkl(path)      -- load pre-computed reference features from PKL
#   download_from_huggingface()   -- download file from HF Hub
#
# ------------------------------------------------------------------------------

import os
import pickle
import logging
import hashlib
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from copy import deepcopy

logger = logging.getLogger(__name__)

# Official NVIDIA CDN URL for Tero's TorchScript inception model.
# This is the same URL used in the original lakonlab/evaluation/metrics.py
# (previously imported from mmgen as TERO_INCEPTION_URL).
TERO_INCEPTION_URL = (
    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/'
    'pretrained/metrics/inception-2015-12-05.pt'
)

# Local directory where downloaded model files are cached.
# Equivalent to mmgen's MMGEN_CACHE_DIR.
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'eval_metrics')

# ---------------------------------------------------------------------------
# InceptionV3 wrapper (torchvision)
# ---------------------------------------------------------------------------

class InceptionV3Features(torch.nn.Module):
    """Wrapper around ``torchvision.models.inception_v3`` that returns both
    pool features (2048-d) and softmax class probabilities (1000-d).

    Input
    -----
    float32 tensor in ``[0, 1]``, shape ``(B, 3, 299, 299)``.
    The model's internal ``transform_input`` handles per-channel normalisation.

    Output
    ------
    ``(feat, pred)`` -- ``feat`` is ``(B, 2048)``, ``pred`` is ``(B, 1000)``.
    """

    def __init__(self):
        super().__init__()
        from torchvision import models
        # Load with pretrained weights; torchvision forces aux_logits=True
        # when loading weights, so we disable it afterwards.
        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
        )
        self.inception.aux_logits = False
        self.inception.AuxLogits = None
        # Store the original fc for computing class logits, then replace it
        # in the model with Identity so the forward returns pool features.
        self._fc = self.inception.fc
        self.inception.fc = torch.nn.Identity()

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 3, 299, 299)``, dtype ``float32``, values in ``[0, 1]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(feat, pred)`` where ``feat`` has shape ``(B, 2048)`` and
            ``pred`` has shape ``(B, 1000)`` (softmax class probabilities).
        """
        feat = self.inception(x)                  # (B, 2048) -- fc is Identity
        logits = self._fc(feat)                   # (B, 1000)
        pred = logits.softmax(dim=1)
        return feat, pred


# ---------------------------------------------------------------------------
# Model caching
# ---------------------------------------------------------------------------

# Module-level dict so each (inception_args, metric) combination is loaded
# at most once per Python process -- matches the _inception_cache in the
# original lakonlab/evaluation/metrics.py.
_inception_cache = {}


def _download_file(url, dest_dir):
    """Download *url* into *dest_dir* and return the local path.

    Parameters
    ----------
    url : str
        HTTPS URL to download.
    dest_dir : str
        Directory in which to save the file.  Created if it does not exist.

    Returns
    -------
    str
        Absolute path to the downloaded (or already-cached) file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(url.split('?')[0])
    dest_path = os.path.join(dest_dir, filename)

    if os.path.isfile(dest_path):
        logger.info(f'Using cached file: {dest_path}')
        return dest_path

    logger.info(f'Downloading {url}')
    logger.info(f'  -> {dest_path}')

    def _reporthook(count, block_size, total_size):
        if total_size > 0 and count % 500 == 0:
            pct = count * block_size * 100 / total_size
            logger.info(f'  {min(pct, 100):.1f}% ...')

    urllib.request.urlretrieve(url, dest_path, reporthook=_reporthook)
    logger.info('  Download complete.')
    return dest_path


def _load_inception_from_path(inception_path, map_location=None):
    """Load Tero's TorchScript inception model from a local ``.pt`` file.

    Parameters
    ----------
    inception_path : str
        Path to the ``.pt`` TorchScript file on disk.
    map_location : optional
        Passed directly to ``torch.jit.load``.  Use ``'cpu'`` to load on CPU
        regardless of where the model was saved.

    Returns
    -------
    torch.nn.Module or None
        The loaded TorchScript module, or ``None`` if loading failed.
    """
    logger.info(f"Loading Tero's Inception model from '{inception_path}'")
    try:
        model = torch.jit.load(inception_path, map_location=map_location)
        logger.info("Tero's Inception model loaded successfully.")
        return model
    except Exception as exc:
        logger.warning(f"Failed to load Tero's Inception model: {exc}")
        return None


def _load_inception_from_url(inception_url, map_location=None):
    """Download Tero's inception model from *inception_url* and load it.

    Parameters
    ----------
    inception_url : str or None
        HTTPS URL of the ``.pt`` file.  Falls back to ``TERO_INCEPTION_URL``
        if ``None`` is passed.
    map_location : optional
        Forwarded to ``_load_inception_from_path``.

    Returns
    -------
    torch.nn.Module or None
        The loaded model, or ``None`` if download/load failed.
    """
    url = inception_url if inception_url else TERO_INCEPTION_URL
    logger.info(f'Downloading Inception model from {url}')
    try:
        path = _download_file(url, CACHE_DIR)
        return _load_inception_from_path(path, map_location=map_location)
    except Exception as exc:
        logger.error(f'Download failed: {exc}')
        return None


def load_inception(inception_args, metric='FID', map_location=None):
    """Load an Inception model with in-process caching.

    Parameters
    ----------
    inception_args : dict
        Configuration dict with the following keys:

        ``type`` (str, default ``'StyleGAN'``)
            ``'StyleGAN'``    -- Tero's TorchScript model (default,
            matches lakonlab's InceptionMetrics).
            ``'InceptionV3'`` -- torchvision InceptionV3 with pretrained
            ImageNet weights.

        ``inception_path`` (str, optional)
            Only used when type is ``'StyleGAN'``:
              - A local filesystem path to an already-downloaded ``.pt`` file.
              - An HTTPS URL; the file will be downloaded automatically.
            Defaults to ``TERO_INCEPTION_URL`` (NVIDIA CDN) when omitted.

    metric : str
        Label used to disambiguate the cache key when the same
        ``inception_args`` dict is reused for different metrics (e.g. ``'FID'``
        vs ``'IS'``).

    map_location : optional
        Forwarded to ``torch.jit.load`` (only relevant for StyleGAN type).

    Returns
    -------
    tuple[torch.nn.Module, str]
        ``(model, style)`` where *style* is ``'StyleGAN'`` or ``'InceptionV3'``.

        For ``'StyleGAN'`` models (Tero's TorchScript, default)::

            feat = model(batch_uint8, return_features=True)
            # batch_uint8 : (B, 3, H, W) uint8 in [0, 255]
            # feat        : (B, 2048) float

        For ``'InceptionV3'`` models (``InceptionV3Features`` wrapper)::

            feat, pred = model(batch_float01)
            # batch_float01 : (B, 3, 299, 299) float32 in [0, 1]
            # feat          : (B, 2048) float -- pool features
            # pred          : (B, 1000) float -- softmax class probs

    Raises
    ------
    TypeError
        If ``inception_args`` is not a dict.
    RuntimeError
        If the StyleGAN model cannot be loaded from path or URL.
    """
    if not isinstance(inception_args, dict):
        raise TypeError(f"'inception_args' must be a dict, got {type(inception_args)}")

    # Stable cache key: md5 of sorted args dict + metric label
    cache_key = (
        hashlib.md5(str(sorted(inception_args.items())).encode()).hexdigest()
        + f'_{metric}'
    )

    if cache_key in _inception_cache:
        return _inception_cache[cache_key]

    _inception_args = deepcopy(inception_args)
    inception_type = _inception_args.pop('type', 'StyleGAN')

    # -- StyleGAN (Tero's TorchScript, default) --------------------------------
    if inception_type == 'StyleGAN':
        path = _inception_args.get('inception_path', TERO_INCEPTION_URL)

        if 'http' not in path:
            model = _load_inception_from_path(path, map_location=map_location)
            if isinstance(model, torch.nn.Module):
                result = model, 'StyleGAN'
                _inception_cache[cache_key] = result
                return result

        model = _load_inception_from_url(path, map_location=map_location)
        if isinstance(model, torch.nn.Module):
            result = model, 'StyleGAN'
            _inception_cache[cache_key] = result
            return result

        raise RuntimeError(
            f'Cannot load StyleGAN Inception with inception_args={inception_args!r}. '
            'Check that the path/URL is valid and the file is a Tero-style .pt.'
        )

    # -- InceptionV3 (torchvision) ---------------------------------------------
    if inception_type == 'InceptionV3':
        logger.info('Loading torchvision InceptionV3 (pretrained) ...')
        model = InceptionV3Features()
        result = model, 'InceptionV3'
        _inception_cache[cache_key] = result
        return result

    raise ValueError(
        f"Unknown inception type '{inception_type}'. "
        "Use 'StyleGAN' (default) or 'InceptionV3'."
    )


# ---------------------------------------------------------------------------
# Hugging Face Hub download helper
# ---------------------------------------------------------------------------

def download_from_huggingface(hf_url):
    """Download a file from Hugging Face Hub.

    Parameters
    ----------
    hf_url : str
        URL in the format ``huggingface://repo_owner/repo_name/path/to/file``.
        For example: ``huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl``

    Returns
    -------
    str
        Local path to the downloaded (cached) file.
    """
    from huggingface_hub import hf_hub_download

    parts = hf_url.replace('huggingface://', '').split('/')
    repo_id = '/'.join(parts[:2])
    repo_filename = '/'.join(parts[2:])
    cached_file = hf_hub_download(repo_id=repo_id, filename=repo_filename)
    return cached_file


# ---------------------------------------------------------------------------
# Reference PKL loading
# ---------------------------------------------------------------------------

def load_reference_pkl(reference_pkl):
    """Load pre-computed inception features from a PKL file.

    The PKL file should contain a dict with keys:
      - ``'mean'``            : np.ndarray (2048,) -- mean of real features
      - ``'cov'``             : np.ndarray (2048, 2048) -- covariance of real features
      - ``'real_feats_np'``   : np.ndarray (N, 2048) -- raw real feature vectors
      - ``'num_real_feeded'`` : int -- number of real images used
      - ``'real_preds_np'``   : np.ndarray (N, C), optional -- softmax class probs for IS

    Parameters
    ----------
    reference_pkl : str
        Path to the PKL file.  Supports:
        - Local filesystem path
        - ``huggingface://`` URL (auto-downloaded via HF Hub)
        - ``http://`` or ``https://`` URL (auto-downloaded)

    Returns
    -------
    dict
        The loaded reference dict with keys ``'mean'``, ``'cov'``,
        ``'real_feats_np'``, ``'num_real_feeded'``.
    """
    # Resolve the path
    if reference_pkl.startswith('huggingface://'):
        reference_pkl = download_from_huggingface(reference_pkl)
    elif reference_pkl.startswith(('http://', 'https://')):
        reference_pkl = _download_file(reference_pkl, CACHE_DIR)

    assert os.path.isfile(reference_pkl), (
        f'Reference PKL file not found: {reference_pkl!r}'
    )

    logger.info(f'Loading reference PKL: {reference_pkl}')
    with open(reference_pkl, 'rb') as f:
        reference = pickle.load(f)

    return reference


# ---------------------------------------------------------------------------
# Generation (fake) features cache: load / save
# Used by FID, IS, PR to avoid re-extracting inception features for generation_dir.
# ---------------------------------------------------------------------------

def load_generation_feats_pkl(pkl_path, num_images=None):
    """Load pre-computed inception features for the generation directory.

    The PKL file should contain a dict with keys:
      - ``'feats_np'``  : np.ndarray (N, 2048) -- pool features (for FID, PR)
      - ``'preds_np'``  : np.ndarray (N, C) -- softmax class probs (for IS)

    Parameters
    ----------
    pkl_path : str
        Local path to the PKL file.
    num_images : int or None, optional
        If set, only the first num_images rows are returned (for consistency
        with args['num_images'] caps).  None means use all.

    Returns
    -------
    dict
        ``{'feats_np': np.ndarray, 'preds_np': np.ndarray}``.  If num_images
        is set, arrays are sliced to that length.
    """
    assert os.path.isfile(pkl_path), (
        f'Generation feats PKL not found: {pkl_path!r}'
    )
    logger.info(f'Loading generation features from {pkl_path}')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    feats_np = data['feats_np']
    preds_np = data['preds_np']
    if num_images is not None:
        feats_np = feats_np[:num_images]
        preds_np = preds_np[:num_images]
    return dict(feats_np=feats_np, preds_np=preds_np)


def save_generation_feats_pkl(pkl_path, feats_np, preds_np):
    """Save inception features and preds for the generation directory.

    FID, IS, and PR can later load this file via load_generation_feats_pkl
    to skip re-extraction.

    Parameters
    ----------
    pkl_path : str
        Local path where the PKL will be written.  Parent directories are
        created if they do not exist.
    feats_np : np.ndarray
        Shape (N, 2048) -- inception pool features.
    preds_np : np.ndarray
        Shape (N, C) -- softmax class probabilities.
    """
    parent = os.path.dirname(pkl_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    logger.info(f'Saving generation features to {pkl_path}')
    with open(pkl_path, 'wb') as f:
        pickle.dump(dict(feats_np=feats_np, preds_np=preds_np), f)


# ---------------------------------------------------------------------------
# Shared image I/O and feature extraction helpers
# (used by FID.py, IS.py, PR.py -- import from here rather than duplicating)
# ---------------------------------------------------------------------------

# Supported image file extensions for _list_images.
_IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def _list_images(directory):
    """Return a sorted list of all image file paths in *directory*.

    Only files whose extension is in ``_IMG_EXTS`` are returned.
    The scan is non-recursive (top-level files only).

    Parameters
    ----------
    directory : str
        Path to the folder to scan.

    Returns
    -------
    list[str]
        Sorted list of absolute image paths found in the directory.
    """
    paths = []
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname)[1].lower() in _IMG_EXTS:
            paths.append(os.path.join(directory, fname))
    return paths


def _load_image_tensor(path):
    """Load a single image as a float32 tensor in [-1, 1].

    The image is loaded via PIL, converted to RGB, and normalised to [-1, 1]
    (matching lakonlab's preprocessing: images arrive as float32 in [-1, 1]).
    No resizing is performed here; resize is handled in extract_inception_features
    when the ``resize`` flag is True.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    torch.Tensor
        Shape ``(3, H, W)``, dtype ``float32``, values in ``[-1, 1]``.
        Always converted to RGB (drops alpha channel if present).
    """
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0      # [0, 1]
    arr = arr * 2.0 - 1.0                               # [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)     # (3, H, W) in [-1, 1]
    return tensor


def extract_inception_features(image_paths, inception_net, inception_style,
                                batch_size=64, device='cpu', resize=False,
                                image_transform=None, pil_transform=None):
    """Extract Inception pool features and class probabilities for a list of images.
    Matches https://github.com/Lakonik/piFlow/blob/67431c35d88779088acec23b38773402ace0fd94/lakonlab/evaluation/metrics.py#L292

    Preprocessing matches lakonlab's ``InceptionMetrics.extract_features``:
      1. Load image as float32 in [-1, 1] (done in ``_load_image_tensor``).
      2. If ``resize=True``, apply ``F.interpolate(bicubic, antialias=True)``
         to 299x299, then clamp to [-1, 1].  (Matches lakonlab lines 708-710.)
      3. Convert to model input format:
         - StyleGAN: ``(x * 127.5 + 128).clamp(0, 255).to(uint8)``
           (matches lakonlab line 712)
         - InceptionV3: ``(x + 1) / 2`` to get [0, 1] range.

    Two model types are supported:

    *StyleGAN* (default) -- Tero's TorchScript model:
      - Input: ``[0, 255]`` uint8.  Model handles internal normalisation.
      - ``feat`` via ``return_features=True``; ``pred`` via output layer.

    *InceptionV3* -- ``InceptionV3Features`` wrapper:
      - Input: ``[0, 1]`` float32.  ``transform_input`` handles normalisation.
      - Returns ``(feat, pred)`` with shapes ``(B, 2048)`` and ``(B, 1000)``.

    Parameters
    ----------
    image_paths : list[str]
        Ordered list of image file paths to process.
    inception_net : torch.nn.Module
        Model returned by ``load_inception``.
    inception_style : str
        ``'StyleGAN'`` or ``'InceptionV3'`` -- returned by ``load_inception``.
    batch_size : int, optional
        Number of images per GPU forward pass.  Default: 64.
    device : str, optional
        Torch device string.  Default: ``'cpu'``.
    resize : bool, optional
        If True, resize images to 299x299 via torch F.interpolate bicubic
        with antialias before feeding to the model.  Default: False.
    image_transform : callable or None, optional
        If provided, called on each batch tensor ``(B, 3, H, W)`` in ``[-1, 1]``
        after loading and before optional resize.  Use this to apply foveation
        masks or spatial downsampling to the images.  Default: None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(feat_np, pred_np)``

        ``feat_np`` -- shape ``(N, 2048)``, dtype float32.
            Pool feature vectors.  Used by **FID** and **PR**.

        ``pred_np`` -- shape ``(N, C)``, dtype float32.
            Softmax class probabilities ``p(y|x)``.  ``C`` = 1008 for
            StyleGAN, 1000 for InceptionV3.  Used by **IS**.
    """
    assert inception_style in ('StyleGAN', 'InceptionV3'), (
        f"inception_style must be 'StyleGAN' or 'InceptionV3', got '{inception_style}'."
    )

    all_feats = []
    all_preds = []
    n = len(image_paths)

    for start in range(0, n, batch_size):
        batch_paths = image_paths[start:start + batch_size]

        # Load each image as PIL, apply optional pil_transform before tensor conversion.
        imgs = []
        for p in batch_paths:
            try:
                img_pil = Image.open(p).convert('RGB')
            except Exception as e:
                logger.warning(f"Skipping image (failed to load): {p} — {e}")
                continue
            if pil_transform is not None:
                img_pil = pil_transform(img_pil)
            arr = np.array(img_pil, dtype=np.float32) / 255.0
            arr = arr * 2.0 - 1.0
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1))

        # Resize per image before stacking so variable-size images can be batched.
        # Must happen here (not after stack) because torch.stack requires equal sizes.
        if resize:
            imgs = [
                F.interpolate(
                    img.unsqueeze(0), size=(299, 299),
                    mode='bicubic', align_corners=False, antialias=True,
                ).clamp(-1, 1).squeeze(0)
                for img in imgs
            ]

        batch = torch.stack(imgs, dim=0).to(device)   # (B, 3, H, W) in [-1,1]

        # Optional image transform (e.g. foveation mask or spatial downsampling).
        if image_transform is not None:
            batch = image_transform(batch)

        with torch.no_grad():
            if inception_style == 'StyleGAN':
                # Convert [-1, 1] -> [0, 255] uint8 (matches lakonlab line 712).
                batch_u8 = (batch * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                feat = inception_net(batch_u8, return_features=True)
                # Compute class probs via the output layer (matches lakonlab line 714).
                _base_net = inception_net.module if hasattr(inception_net, 'module') else inception_net
                pred = F.linear(feat, _base_net.output.weight).softmax(dim=1)
            else:
                # InceptionV3Features expects float32 [0, 1].
                batch_01 = (batch + 1.0) / 2.0   # [-1,1] -> [0,1]
                feat, pred = inception_net(batch_01)

        all_feats.append(feat.cpu().float())
        all_preds.append(pred.cpu().float())

        end = min(start + batch_size, n)
        if end % (batch_size * 10) == 0 or end == n:
            logger.info(f'  {end}/{n} images processed')

    feat_np = torch.cat(all_feats, dim=0).numpy()
    pred_np = torch.cat(all_preds, dim=0).numpy()
    return feat_np, pred_np
