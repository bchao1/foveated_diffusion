"""HPSV2.py -- HPS v2 (Human Preference Score v2) evaluation.

Computes the HPSv2 score for a list of (image, prompt) pairs using a
single model load and batched GPU inference.

Deferred imports: `torch` and `hpsv2` are imported inside the function
body so that callers can set CUDA_VISIBLE_DEVICES before any CUDA
initialization occurs.

API usage
---------
    from metric_computation.HPSV2 import compute_hpsv2
    result = compute_hpsv2(image_paths, prompts, batch_size=32)
    # result = {'hpsv2': float, 'scores': list[float]}
"""

import logging
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_hpsv2(image_paths, prompts, batch_size=64, hps_version='v2.1', desc='HPSv2'):
    """Compute the mean HPSv2 score for a list of (image, prompt) pairs.

    Loads the HPSv2 model checkpoint once and runs batched GPU inference.
    torch and hpsv2 are imported here (not at module level) so that the
    caller can set CUDA_VISIBLE_DEVICES before any CUDA initialisation.

    Parameters
    ----------
    image_paths : list[str]
        Paths to image files.
    prompts : list[str]
        Prompt strings, one per image (same order as image_paths).
    batch_size : int
        Number of images per forward pass.
    hps_version : str
        HPS model version string, e.g. 'v2.0' or 'v2.1'.
    desc : str
        tqdm progress-bar description.

    Returns
    -------
    dict
        ``'hpsv2'``  (float)       -- mean HPS score across all images.
        ``'scores'`` (list[float]) -- per-image scores (for correct weighted
                                      averaging when merging partial results).
    """
    import torch
    import hpsv2  # noqa: F401  (ensures package is importable; internals used below)
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from hpsv2.utils import hps_version_map
    from huggingface_hub import hf_hub_download

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ------------------------------------------------------------------
    # Load model + checkpoint once
    # ------------------------------------------------------------------
    model, _, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        'laion2b_s32b_b79k',
        precision='amp',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )

    checkpoint_path = hf_hub_download('xswu/HPSv2', hps_version_map[hps_version])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    tokenizer = get_tokenizer('ViT-H-14')

    # ------------------------------------------------------------------
    # Batched scoring loop
    # ------------------------------------------------------------------
    n = len(image_paths)
    all_scores = []

    for start in tqdm(range(0, n, batch_size), desc=desc):
        batch_paths = image_paths[start:start + batch_size]
        batch_prompts = prompts[start:start + batch_size]

        imgs = [preprocess_val(Image.open(p).convert('RGB')) for p in batch_paths]

        imgs_t = torch.stack(imgs).to(device)
        text_t = tokenizer(batch_prompts).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            out = model(imgs_t, text_t)
            scores = (out['image_features'] * out['text_features']).sum(dim=1)

        all_scores.extend(scores.float().cpu().tolist())

    return {'hpsv2': float(np.mean(all_scores)), 'scores': all_scores}
