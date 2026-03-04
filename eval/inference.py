"""
FLUX2 Foveated Image Generation - DiffSynth-based Inference Script

This script demonstrates foveated image generation using the DiffSynth framework
with TokenUpsampler and TokenDownsampler for high-quality mixed-resolution rendering.

Experiments:
- baseline: Full resolution generation without foveation
- circular: Rotating circular foveation mask (creates a video-like sequence)
- radius_ablation: Test different foveation mask radii
- position_ablation: Test different foveation mask positions
"""

import argparse
import os
import sys
import math
import time
import random
import subprocess
import yaml

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../image_generation"))

from diffsynth.pipelines.flux2_image_foveated import Flux2FoveatedImagePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.core import load_state_dict

import pandas as pd

# Import Token Upsampler/Downsampler from image_generation (adapter and dit arch)
try:
    from adapter_foveation_time_cond import (
        TokenDownsampler,
        TokenUpsampler,
        TokenDiTUpsampler,
        TokenDiTDownsampler,
    )
except ImportError:
    TokenUpsampler = None
    TokenDownsampler = None
    TokenDiTUpsampler = None
    TokenDiTDownsampler = None

# Import Flux2 Token AE (flux2 arch)
try:
    from diffsynth.models.flux2_token_ae import Flux2TokenDownsampler, Flux2TokenUpsampler
except ImportError:
    Flux2TokenDownsampler = None
    Flux2TokenUpsampler = None

# Config keys saved by train_tokenAE_pretrain.py (for loading tokenAE_config.yaml)
TOKEN_AE_CONFIG_KEYS = {
    "tokenAE_arch",
    "in_channels",
    "feature_channels",
    "embedding_dim",
    "upsampler_num_res_blocks",
    "downsampler_num_res_blocks",
    "tokenAE_dont_use_learned_components",
    "upsampler_normalization_type",
    "downsampler_normalization_type",
    "upsampler_variance_correction",
    "downsampler_variance_correction",
    "naive_upsample_mode",
    "naive_downsample_mode",
    "flux2_naive_downsample_mode",
    "flux2_resnet_eps",
    "flux2_resnet_act_fn",
    "flux2_resnet_groups",
    "flux2_downsampler_add_norm_act_before_conv_out",
    "flux2_upsampler_add_norm_act_before_conv_out",
    "flux2_upsampler_num_pre_upsampler_res_blocks",
    "flux2_upsampler_naive_upsample_mode",
    "flux2_downsampler_add_attention",
    "flux2_downsampler_attention_head_dim",
    "flux2_downsampler_attn_groups",
    "flux2_upsampler_add_attention",
    "flux2_upsampler_attention_head_dim",
    "flux2_upsampler_attn_groups",
    # DiT token up/down sampler (tokenAE_arch=dit)
    "upsampler_num_dit_blocks",
    "downsampler_num_dit_blocks",
    "upsampler_dit_use_temb",
    "downsampler_dit_use_temb",
}


def create_foveation_mask(
    height: int,
    width: int,
    center: tuple = (0.0, 0.0),
    r: float = 0.25,
    shape: str = "circular",
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a low-res foveation mask and upsample to token grid size with nearest interpolation.
    Returns a float mask of shape (height//16, width//16) ready for the pipeline.

    Args:
        height, width (int): Full image dimensions (mask is created at token_grid/2 then upsampled).
        center (tuple): (x, y) normalized coordinates in [-0.5, 0.5].
        r (float): For circular: radius relative to half the diagonal. For square: ratio of side to
            image (1 = full image, 0 = all black). Ignored for checkerboard.
        shape (str): "circular", "square", or "checkerboard" (interleaved pattern).
        device: Device for the tensor (default: cpu).

    Returns:
        torch.Tensor: Float mask of shape (height//16, width//16), values 0 or 1.
    """
    if device is None:
        device = torch.device("cpu")
    high_h, high_w = height // 16, width // 16
    if shape == "checkerboard":
        i = torch.arange(high_h, device=device, dtype=torch.long)
        j = torch.arange(high_w, device=device, dtype=torch.long)
        ii, jj = torch.meshgrid(i, j, indexing="ij")
        return ((ii + jj) % 2 == 0).float()
    low_h, low_w = high_h // 2, high_w // 2
    y_range = torch.arange(low_h, device=device, dtype=torch.float32)
    x_range = torch.arange(low_w, device=device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")
    cx = int((center[0] + 0.5) * low_w)
    cy = int((center[1] + 0.5) * low_h)

    if shape == "square":
        # r = ratio to image size: 1 = full, 0 = all black. Side length = r * min(low_h, low_w)
        side = r * min(low_h, low_w)
        half = side / 2.0
        in_x = (x_grid >= cx - half) & (x_grid <= cx + half)
        in_y = (y_grid >= cy - half) & (y_grid <= cy + half)
        mask_low = (in_x & in_y).unsqueeze(0).unsqueeze(0).float()
    else:
        # circular: r = radius relative to half diagonal
        diagonal = (low_h**2 + low_w**2) ** 0.5
        radius_px = r * (diagonal / 2.0)
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        mask_low = (dist_sq <= radius_px**2).unsqueeze(0).unsqueeze(0).float()

    mask_high = F.interpolate(mask_low, size=(high_h, high_w), mode="nearest")
    return mask_high.squeeze(0).squeeze(0)


def create_foveation_mask_full_res(
    height: int,
    width: int,
    center: tuple = (0.0, 0.0),
    r: float = 0.25,
    shape: str = "circular",
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a foveation mask at full decoded image size (height, width) with the same
    geometry (center, r, shape) as the token-grid mask. Used for merge decode when
    passing full_res_foveation_mask to the pipeline.
    Returns a float mask of shape (height, width), values 0 or 1.
    """
    if device is None:
        device = torch.device("cpu")
    y_range = torch.arange(height, device=device, dtype=torch.float32)
    x_range = torch.arange(width, device=device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")
    cx = (center[0] + 0.5) * width
    cy = (center[1] + 0.5) * height

    if shape == "checkerboard":
        i = torch.arange(height, device=device, dtype=torch.long)
        j = torch.arange(width, device=device, dtype=torch.long)
        ii, jj = torch.meshgrid(i, j, indexing="ij")
        return ((ii + jj) % 2 == 0).float()
    if shape == "square":
        side = r * min(height, width)
        half = side / 2.0
        in_x = (x_grid >= cx - half) & (x_grid <= cx + half)
        in_y = (y_grid >= cy - half) & (y_grid <= cy + half)
        return (in_x & in_y).float()
    # circular
    diagonal = (height ** 2 + width ** 2) ** 0.5
    radius_px = r * (diagonal / 2.0)
    dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    return (dist_sq <= radius_px ** 2).float()


def _polygon_vertices(shape_name: str, scale: float, center=(0.0, 0.0)):
    """
    Vertices of a polygon in normalized coords [-0.5, 0.5], centered at center.
    scale: half-diagonal radius (same meaning as circular r).
    shape_name: "triangle", "square", "hexagon", "star".
    Returns list of (x, y) in normalized coordinates.
    """
    cx, cy = center
    if shape_name == "triangle":
        n = 3
        angles = [2 * math.pi * (i / n) - math.pi / 2 for i in range(n)]  # one vertex up
        return [(cx + scale * math.cos(a), cy + scale * math.sin(a)) for a in angles]
    if shape_name == "square":
        n = 4
        angles = [2 * math.pi * (i / n) - math.pi / 4 for i in range(n)]
        return [(cx + scale * math.cos(a), cy + scale * math.sin(a)) for a in angles]
    if shape_name == "hexagon":
        n = 6
        angles = [2 * math.pi * (i / n) - math.pi / 6 for i in range(n)]
        return [(cx + scale * math.cos(a), cy + scale * math.sin(a)) for a in angles]
    if shape_name == "star":
        # 5-pointed star: 5 outer + 5 inner (inner at scale * 0.4, rotated by pi/5)
        outer = [2 * math.pi * (i / 5) - math.pi / 2 for i in range(5)]
        inner = [2 * math.pi * (i / 5) - math.pi / 2 + math.pi / 5 for i in range(5)]
        verts = []
        for i in range(5):
            verts.append((cx + scale * math.cos(outer[i]), cy + scale * math.sin(outer[i])))
            verts.append((cx + scale * 0.4 * math.cos(inner[i]), cy + scale * 0.4 * math.sin(inner[i])))
        return verts
    raise ValueError("shape_name must be triangle, square, hexagon, or star")


def _point_in_polygon(x: float, y: float, vertices: list) -> bool:
    """Ray-casting: odd number of crossings with polygon edges => inside."""
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
        j = i
    return inside


def _rasterize_polygon_mask(height: int, width: int, vertices_norm: list, device) -> torch.Tensor:
    """
    Rasterize a polygon (vertices in normalized coords [-0.5, 0.5]) to a binary mask (height, width).
    Uses matplotlib.path.Path if available for speed; otherwise fallback to per-pixel ray casting.
    """
    y_range = torch.arange(height, device=device, dtype=torch.float32)
    x_range = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")
    x_norm = (xx + 0.5) / width - 0.5
    y_norm = (yy + 0.5) / height - 0.5
    try:
        from matplotlib.path import Path
        p = Path(vertices_norm)
        xy = np.stack([x_norm.cpu().numpy().ravel(), y_norm.cpu().numpy().ravel()], axis=1)
        inside = p.contains_points(xy)
        mask = torch.from_numpy(inside.astype(np.float32)).reshape(height, width).to(device)
    except ImportError:
        mask = torch.zeros(height, width, device=device, dtype=torch.float32)
        for i in range(height):
            for j in range(width):
                if _point_in_polygon(float(x_norm[i, j]), float(y_norm[i, j]), vertices_norm):
                    mask[i, j] = 1.0
    return mask


def _create_polygon_mask_trajectory(height: int, width: int, shape_name: str, center: tuple, scale: float, device):
    """Create token-grid and full-res masks for a single polygon. scale in [0,1] like circular r (relative to half-diagonal)."""
    vertices = _polygon_vertices(shape_name, scale, center=center)
    full_res = _rasterize_polygon_mask(height, width, vertices, device)
    high_h, high_w = height // 16, width // 16
    token_mask = F.interpolate(
        full_res.unsqueeze(0).unsqueeze(0), size=(high_h, high_w), mode="nearest"
    ).squeeze(0).squeeze(0)
    return token_mask, full_res


def _sample_nonoverlapping_circles(rng, num_circles: int, r_lo: float, r_hi: float, gap: float = 0.02):
    """
    Sample num_circles (2 or 3) circles that are non-overlapping and distributed across
    the image. Uses fixed zones (quadrants + center) so centers are spread; then
    assigns radii and shrinks if needed to enforce non-overlap.
    Returns list of (cx_norm, cy_norm, r) in normalized coords; r relative to half-diagonal.
    """
    # Zones (x_min, x_max, y_min, y_max) in normalized coords to spread centers across the image
    zones = [
        (-0.42, -0.12, -0.42, -0.12),   # top-left
        (0.12, 0.42, -0.42, -0.12),     # top-right
        (-0.42, -0.12, 0.12, 0.42),     # bottom-left
        (0.12, 0.42, 0.12, 0.42),       # bottom-right
        #(-0.18, 0.18, -0.18, 0.18),     # center
    ]
    zone_order = rng.permutation(len(zones))
    chosen_zones = [zone_order[i] for i in range(num_circles)]
    centers = []
    for zi in chosen_zones:
        xmin, xmax, ymin, ymax = zones[zi]
        cx = float(rng.uniform(xmin, xmax))
        cy = float(rng.uniform(ymin, ymax))
        centers.append((cx, cy))
    # Sample radii (in half-diagonal units). Convert to normalized distance for check:
    # half-diagonal in norm coords = sqrt(2)/2 for unit square, so r_norm = r * sqrt(2)/2.
    radii = [float(rng.uniform(r_lo, r_hi)) for _ in range(num_circles)]
    r_to_norm = math.sqrt(2) / 2.0
    gap_norm = gap
    for _ in range(30):
        overlap = False
        for i in range(num_circles):
            for j in range(i + 1, num_circles):
                cix, ciy = centers[i]
                cjx, cjy = centers[j]
                d_norm = math.sqrt((cix - cjx) ** 2 + (ciy - cjy) ** 2)
                need_norm = (radii[i] + radii[j]) * r_to_norm + gap_norm
                if d_norm < need_norm and d_norm > 1e-6:
                    overlap = True
                    # shrink radii so (r_i + r_j) * r_to_norm + gap = d_norm
                    max_sum_r = (d_norm - gap_norm) / r_to_norm
                    scale = max_sum_r / (radii[i] + radii[j])
                    radii[i] = min(radii[i], radii[i] * scale)
                    radii[j] = min(radii[j], radii[j] * scale)
                elif d_norm < 1e-6:
                    radii[i] = r_lo
                    radii[j] = r_lo
                    overlap = True
        if not overlap:
            break
    return [(centers[i][0], centers[i][1], max(r_lo, radii[i])) for i in range(num_circles)]


def _create_multi_circle_mask(height: int, width: int, circles: list, device) -> torch.Tensor:
    """
    circles: list of (cx_norm, cy_norm, r) in normalized coords and r relative to half-diagonal.
    Returns binary mask (height, width) as union of circles.
    """
    y_range = torch.arange(height, device=device, dtype=torch.float32)
    x_range = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")
    diagonal = (height ** 2 + width ** 2) ** 0.5
    mask = torch.zeros(height, width, device=device, dtype=torch.float32)
    for (cx_norm, cy_norm, r) in circles:
        cx = (cx_norm + 0.5) * width
        cy = (cy_norm + 0.5) * height
        radius_px = r * (diagonal / 2.0)
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = torch.maximum(mask, (dist_sq <= radius_px ** 2).float())
    return mask


def generate_foveation_trajectory_masks(height, width, args, device):
    """
    Generate a list of foveation masks for the trajectory grid.
    Uses args.foveation_trajectory_type:
      - "radius": vary r at center
      - "circular": center on orbit
      - "polygons": sequence of triangle/square/hexagon/star with random center and scale
      - "multi_circle": each mask is 2-3 disjoint circles with varying position and size
    Same trajectory is reused for all prompts.
    Returns (masks, full_res_masks) where full_res_masks are at (height, width) with same geometry.
    """
    num_cols = getattr(args, "num_cols", 4)
    orbit_radius = getattr(args, "orbit_radius", 0.25)
    mask_radius = getattr(args, "mask_radius", 0.30)
    traj_type = getattr(args, "foveation_trajectory_type", "circular")

    masks = []
    full_res_masks = []
    rng = np.random.default_rng(getattr(args, "seed", 0))

    if traj_type == "radius":
        r_list = np.linspace(0.3, 0.7, num_cols)
        for col in range(num_cols):
            r = float(r_list[col])
            m = create_foveation_mask(
                height, width, center=(0, 0), r=r,
                shape=getattr(args, "mask_shape", "circular"), device=device
            )
            m_full = create_foveation_mask_full_res(
                height, width, center=(0, 0), r=r,
                shape=getattr(args, "mask_shape", "circular"), device=device
            )
            masks.append(m)
            full_res_masks.append(m_full)
    elif traj_type == "polygons":
        # Sequence of masks with arbitrary sampled shapes: triangle, square, hexagon, 5-pointed star
        shape_choices = ["triangle", "square", "hexagon", "star"]
        center_range = getattr(args, "polygon_center_range", 0.25)
        scale_lo = 0.3 #getattr(args, "polygon_scale_lo", 0.3)
        scale_hi = 0.4 #getattr(args, "polygon_scale_hi", 0.4)
        for col in range(num_cols):
            shape_name = shape_choices[col % len(shape_choices)]
            cx = float(rng.uniform(-center_range, center_range))
            cy = float(rng.uniform(-center_range, center_range))
            scale = float(rng.uniform(scale_lo, scale_hi))
            m, m_full = _create_polygon_mask_trajectory(
                height, width, shape_name, (cx, cy), scale, device
            )
            masks.append(m)
            full_res_masks.append(m_full)
    elif traj_type == "multi_circle":
        # Sequence of masks; each mask is 2-3 non-overlapping circles distributed across the image
        num_circles_lo = 2
        num_circles_hi = 3  # 2 or 3
        r_lo = 0.3 #getattr(args, "multi_circle_r_lo", 0.2)
        r_hi = 0.3 #getattr(args, "multi_circle_r_hi", 0.4)
        gap = 0.3 #getattr(args, "multi_circle_gap", 0.02)
        for col in range(num_cols):
            num_circles = int(rng.integers(num_circles_lo, num_circles_hi))
            circles = _sample_nonoverlapping_circles(rng, num_circles, r_lo, r_hi, gap=gap)
            m_full = _create_multi_circle_mask(height, width, circles, device)
            high_h, high_w = height // 16, width // 16
            m = F.interpolate(
                m_full.unsqueeze(0).unsqueeze(0), size=(high_h, high_w), mode="nearest"
            ).squeeze(0).squeeze(0)
            masks.append(m)
            full_res_masks.append(m_full)
    else:
        # circular: mask center moves on orbit
        angles = [2 * math.pi * (c / num_cols) for c in range(num_cols)]
        for col in range(num_cols):
            angle = angles[col]
            center = (orbit_radius * math.cos(angle), orbit_radius * math.sin(angle))
            m = create_foveation_mask(
                height, width, center=center, r=mask_radius,
                shape=getattr(args, "mask_shape", "circular"), device=device
            )
            m_full = create_foveation_mask_full_res(
                height, width, center=center, r=mask_radius,
                shape=getattr(args, "mask_shape", "circular"), device=device
            )
            masks.append(m)
            full_res_masks.append(m_full)
    return masks, full_res_masks


def make_videos(folder, output_file, fps=5):
    """Combine generated images and masks into a video."""
    try:
        import imageio
        import cv2
    except ImportError:
        print("Warning: imageio or cv2 not installed, skipping video creation")
        return
    
    image_files = [f for f in sorted(os.listdir(folder)) if f.startswith("img") and f.endswith('.png')]
    mask_files = [f for f in sorted(os.listdir(folder)) if f.startswith("foveation_mask") and f.endswith('.png')]
    image_files.sort()
    mask_files.sort()
    
    if len(image_files) == 0:
        print("No images found for video creation")
        return
        
    print(f"Creating video with {len(image_files)} images and {len(mask_files)} masks")
    images = [imageio.imread(os.path.join(folder, f)) for f in image_files]
    masks = [imageio.imread(os.path.join(folder, f)) for f in mask_files] if mask_files else [None] * len(images)
    
    with imageio.get_writer(output_file, fps=fps) as writer:
        for image, mask in zip(images, masks):
            if mask is not None:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                combined = np.concatenate([image, mask], axis=1)
            else:
                combined = image
            writer.append_data(combined)
    
    print(f"Video saved to {output_file}")

def load_prompt_dataset(prompt_dataset_path):
    """Load the prompt dataset from a csv file.
    Returns a list of prompts.
    csv file has the following columns: image,prompt
    """
    prompts = []
    df = pd.read_csv(prompt_dataset_path)
    for index, row in df.iterrows():
        prompts.append(row['prompt'])
    return prompts


def _gaussian_blur_mask_2d(mask_4d, sigma_pixels, device, dtype):
    """Apply separable Gaussian blur to [1, 1, H, W] mask. Matches pipeline soft blend."""
    k = max(3, int(math.ceil(3 * sigma_pixels)) * 2 + 1)
    if k % 2 == 0:
        k += 1
    x = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma_pixels ** 2 + 1e-6))
    g = g / g.sum()
    kernel_h = g.view(1, 1, k, 1)
    kernel_w = g.view(1, 1, 1, k)
    pad = k // 2
    out = F.conv2d(mask_4d, kernel_h, padding=(pad, 0))
    out = F.conv2d(out, kernel_w, padding=(0, pad))
    return out


def draw_foveation_outline(
    image_np,
    mask_token_grid,
    height,
    width,
    outline_width_frac=0.01,
    color=(255, 0, 0),
):
    """
    Draw the foveation region outline on an image. Mask is at token grid resolution (height//16, width//16).
    Outline is drawn as: outer black stroke (10% of border width), then inner white stroke (80% of border width).

    Args:
        image_np: (H, W, C) uint8 RGB numpy image.
        mask_token_grid: (height//16, width//16) float 0/1 mask from create_foveation_mask.
        height, width: Full image dimensions (must match image_np shape).
        outline_width_frac: Total border width as fraction of image width (default 0.01 = 1%).
        color: Unused (kept for API compatibility); outline is always black + white.

    Returns:
        image_np with outline drawn in-place (same array).
    """
    try:
        import cv2
    except ImportError:
        return image_np
    mask_np = mask_token_grid.detach().cpu().numpy() if hasattr(mask_token_grid, "detach") else np.asarray(mask_token_grid)
    mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
    mask_high = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_width = max(1, int(outline_width_frac * width))
    black_stroke = max(1, int(0.10 * border_width))
    white_stroke = max(1, int(0.80 * border_width))
    # Draw outer black first, then inner white so black is 10% outside and white is 80% inside
    cv2.drawContours(image_np, contours, -1, (255, 255, 255), black_stroke + white_stroke)
    cv2.drawContours(image_np, contours, -1, (255, 255, 255), white_stroke)
    return image_np


def load_pipeline(args, use_foveated_pipeline=True):
    """Load the FLUX2 Foveated Pipeline."""
    print("Loading FLUX2 Foveated Pipeline...")
    pipeline_class = Flux2FoveatedImagePipeline if use_foveated_pipeline else Flux2ImagePipeline
    pipe = pipeline_class.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id=args.model_id, origin_file_pattern="tokenizer/"),
    )
    
    return pipe


def _load_token_ae_config_from_folder(folder):
    """Load tokenAE_config.yaml from checkpoint folder. Returns {} if not found."""
    if folder is None:
        return {}
    for name in ("tokenAE_config.yaml", "tokenAE_config.yml"):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    cfg = yaml.safe_load(f)
                return cfg if isinstance(cfg, dict) else {}
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
                return {}
    return {}


def _get_cfg(cfg, args, key, default=None):
    """Get value from config, else args, else default."""
    if cfg is not None and key in cfg:
        return cfg[key]
    return getattr(args, key, default)


def _str_to_bool(s):
    """Parse true/false to bool. Accepts bool (passthrough), 'true'/'false'/'1'/'0'/'yes'/'no', or numeric."""
    if isinstance(s, bool):
        return s
    if isinstance(s, str):
        return s.lower() in ("true", "1", "yes")
    if isinstance(s, (int, float)):
        return bool(s)
    raise argparse.ArgumentTypeError("Expected true/false (or bool), got {!r}".format(s))


def count_token_ae_params(token_upsampler, token_downsampler):
    """Count and print total parameters of token upsampler and downsampler."""
    def _count(m):
        return sum(p.numel() for p in m.parameters()) if m is not None else 0
    up_params = _count(token_upsampler)
    down_params = _count(token_downsampler)
    total = up_params + down_params
    print(f"TokenAE params: upsampler={up_params:,}, downsampler={down_params:,}, total={total:,}")


def load_token_ae(args, device):
    """Load TokenUpsampler and TokenDownsampler. Supports adapter and flux2 architectures.
    Uses tokenAE_config.yaml from tokenAE_folder when present; otherwise argparse defaults."""
    if not args.tokenAE_training:
        return None, None
    # Load config from folder if available
    saved_cfg = _load_token_ae_config_from_folder(args.tokenAE_folder)
    if saved_cfg:
        print(f"Using token AE config from {args.tokenAE_folder}/tokenAE_config.yaml")
        # Apply saved config to args for consistent getattr
        for key in TOKEN_AE_CONFIG_KEYS:
            if key in saved_cfg:
                setattr(args, key, saved_cfg[key])

    arch = getattr(args, "tokenAE_arch", "adapter")

    if arch == "flux2":
        if Flux2TokenDownsampler is None or Flux2TokenUpsampler is None:
            print("Flux2TokenAE not found (diffsynth.models.flux2_token_ae). Using base model.")
            return None, None
        # Build config dict for from_config (saved_cfg + args fallback)
        cfg = dict(saved_cfg) if saved_cfg else {}
        for key in TOKEN_AE_CONFIG_KEYS:
            if key not in cfg and hasattr(args, key):
                cfg[key] = getattr(args, key)
        token_upsampler = Flux2TokenUpsampler.from_config(cfg)
        token_downsampler = Flux2TokenDownsampler.from_config(cfg)
        print(f"Created Flux2 TokenAE (in_channels={cfg.get('in_channels', 32)})")
    elif arch == "dit":
        if TokenDiTUpsampler is None or TokenDiTDownsampler is None:
            print("TokenDiTUpsampler/TokenDiTDownsampler not found (adapter_foveation_time_cond). Using base model.")
            return None, None
        in_channels = _get_cfg(saved_cfg, args, "in_channels", 32)
        up_num_dit_blocks = _get_cfg(saved_cfg, args, "upsampler_num_dit_blocks", 1)
        down_num_dit_blocks = _get_cfg(saved_cfg, args, "downsampler_num_dit_blocks", 1)
        up_use_temb = _str_to_bool(_get_cfg(saved_cfg, args, "upsampler_dit_use_temb", False))
        down_use_temb = _str_to_bool(_get_cfg(saved_cfg, args, "downsampler_dit_use_temb", False))
        naive_upsample_mode = _get_cfg(saved_cfg, args, "naive_upsample_mode", "bilinear")
        naive_downsample_mode = _get_cfg(saved_cfg, args, "naive_downsample_mode", "bilinear")
        token_upsampler = TokenDiTUpsampler(
            in_channels=in_channels,
            num_dit_blocks=up_num_dit_blocks,
            use_temb=up_use_temb,
            naive_upsample_mode=naive_upsample_mode,
            use_as_refiner=args.use_as_refiner,
        )
        token_downsampler = TokenDiTDownsampler(
            in_channels=in_channels,
            num_dit_blocks=down_num_dit_blocks,
            use_temb=down_use_temb,
            naive_downsample_mode=naive_downsample_mode,
        )
        print(f"Created DiT TokenAE (up_num_dit_blocks={up_num_dit_blocks}, down_num_dit_blocks={down_num_dit_blocks}, up_use_temb={up_use_temb}, down_use_temb={down_use_temb})")
    else:
        if TokenUpsampler is None or TokenDownsampler is None:
            print("TokenUpsampler/TokenDownsampler not found (adapter_foveation_time_cond). Using base model.")
            return None, None
        in_channels = _get_cfg(saved_cfg, args, "in_channels", 32)
        feature_channels = _get_cfg(saved_cfg, args, "feature_channels", 64)
        embedding_dim = _get_cfg(saved_cfg, args, "embedding_dim", 3072)
        upsampler_num_res_blocks = _get_cfg(saved_cfg, args, "upsampler_num_res_blocks", 10)
        downsampler_num_res_blocks = _get_cfg(saved_cfg, args, "downsampler_num_res_blocks", 1)
        tokenAE_dont_use_learned_components = _get_cfg(saved_cfg, args, "tokenAE_dont_use_learned_components", False)
        upsampler_normalization_type = _get_cfg(saved_cfg, args, "upsampler_normalization_type", "gn")
        downsampler_normalization_type = _get_cfg(saved_cfg, args, "downsampler_normalization_type", "none")
        upsampler_variance_correction = _get_cfg(saved_cfg, args, "upsampler_variance_correction")
        downsampler_variance_correction = _get_cfg(saved_cfg, args, "downsampler_variance_correction")
        naive_upsample_mode = _get_cfg(saved_cfg, args, "naive_upsample_mode", "nearest")
        naive_downsample_mode = _get_cfg(saved_cfg, args, "naive_downsample_mode", "nearest")
        token_upsampler = TokenUpsampler(
            in_channels=in_channels,
            feature_channels=feature_channels,
            embedding_dim=embedding_dim,
            num_res_blocks=upsampler_num_res_blocks,
            naive_upsample_mode=naive_upsample_mode,
            learn_residual=True,
            variance_correction=upsampler_variance_correction,
            normalization_type=upsampler_normalization_type,
            use_learned_components=not tokenAE_dont_use_learned_components,
        )
        token_downsampler = TokenDownsampler(
            in_channels=in_channels,
            feature_channels=feature_channels,
            embedding_dim=embedding_dim,
            num_res_blocks=downsampler_num_res_blocks,
            naive_downsample_mode=naive_downsample_mode,
            learn_residual=True,
            variance_correction=downsampler_variance_correction,
            normalization_type=downsampler_normalization_type,
            use_learned_components=not tokenAE_dont_use_learned_components,
        )
        if not saved_cfg:
            print("Creating adapter TokenAE from argparse defaults (no tokenAE_config.yaml found)")

    if args.tokenAE_folder is not None:
        up_path = os.path.join(args.tokenAE_folder, "token_upsampler.pt")
        down_path = os.path.join(args.tokenAE_folder, "token_downsampler.pt")
        if os.path.isfile(up_path) and os.path.isfile(down_path):
            print(f"Loading TokenAE weights from {args.tokenAE_folder}")
            token_upsampler.load_state_dict(
                torch.load(up_path, map_location="cpu", weights_only=False)
            )
            token_downsampler.load_state_dict(
                torch.load(down_path, map_location="cpu", weights_only=False)
            )
        else:
            print("Warning: token_upsampler.pt or token_downsampler.pt not found, using untrained weights.")
    else:
        print("Not loading TokenAE checkpoints, using untrained weights.")

    token_upsampler = token_upsampler.to(device).to(torch.bfloat16)
    token_downsampler = token_downsampler.to(device).to(torch.bfloat16)
    return token_upsampler, token_downsampler

# single-prompt generation (foveation_mask passed in; high_res ignores it)
def run_high_res(pipe, args, output_dir, prompt, prompt_idx, token_upsampler=None, token_downsampler=None, foveation_mask=None, full_res_foveation_mask=None):
    """Run baseline full-resolution generation."""
    print(f"Running high-res generation for prompt {prompt_idx:010d}: {prompt}")
    image = pipe(
        prompt=prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        rand_device="cuda",
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.guidance_scale,
        foveation_mask=None,
        prediction_type=args.prediction_type,
        tokenAE_training=False,
        decode_mode="direct"
    )
    image.save(os.path.join(output_dir, f"img_{prompt_idx:010d}.png"))

def run_naive_mixed_res(pipe, args, output_dir, prompt, prompt_idx, token_upsampler, token_downsampler, foveation_mask, full_res_foveation_mask=None):
    """Run naive mixed-resolution generation. foveation_mask must be provided."""
    print(f"Running naive mixed-resolution generation for prompt {prompt_idx:010d}: {prompt}")
    pipe_kwargs = dict(
        prompt=prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        rand_device="cuda",
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.guidance_scale,
        foveation_mask=foveation_mask,
        token_upsampler=token_upsampler,
        token_downsampler=token_downsampler,
        decode_mode=args.decode_mode,
        prediction_type=args.prediction_type,
        tokenAE_training=args.tokenAE_training,
        soft_foveation_blend=getattr(args, "soft_foveation_blend", False),
    )
    if full_res_foveation_mask is not None:
        pipe_kwargs["full_res_foveation_mask"] = full_res_foveation_mask
    image = pipe(**pipe_kwargs)
    image.save(os.path.join(output_dir, f"img_{prompt_idx:010d}.png"))

def run_ours(pipe, args, output_dir, prompt, prompt_idx, token_upsampler, token_downsampler, foveation_mask, full_res_foveation_mask=None):
    """Run our proposed method. foveation_mask must be provided."""
    print(f"Running our proposed method for prompt {prompt_idx:010d}: {prompt}")
    pipe_kwargs = dict(
        prompt=prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        rand_device="cuda",
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.guidance_scale,
        foveation_mask=foveation_mask,
        token_upsampler=token_upsampler,
        token_downsampler=token_downsampler,
        decode_mode=args.decode_mode,
        prediction_type=args.prediction_type,
        tokenAE_training=args.tokenAE_training,
        soft_foveation_blend=getattr(args, "soft_foveation_blend", False),
    )
    if full_res_foveation_mask is not None:
        pipe_kwargs["full_res_foveation_mask"] = full_res_foveation_mask
    image = pipe(**pipe_kwargs)
    image.save(os.path.join(output_dir, f"img_{prompt_idx:010d}.png"))

SINGLE_PROMPT_GENERATION_FUNCTIONS = {
    "high_res": run_high_res,
    "naive_mixed_res": run_naive_mixed_res,
    "ours": run_ours,
}

# full-prompt generation wrapper. calls the single-prompt generation function for each prompt.
def run_full_eval(single_prompt_generation_func, pipe, args, output_dir, prompts, token_upsampler=None, token_downsampler=None):
    """Run full-prompt generation. Passes foveation mask for naive/ours (shape from args.full_eval_mask); None for high_res."""
    default_mask = None
    full_res_mask = None
    if single_prompt_generation_func is not run_high_res:
        mask_shape = getattr(args, "full_eval_mask", "square")
        r = getattr(args, "mask_radius", 0.5)
        default_mask = create_foveation_mask(
            args.height, args.width, center=(0, 0), r=r, shape=mask_shape, device=pipe.device
        )
        full_res_mask = create_foveation_mask_full_res(
            args.height, args.width, center=(0, 0), r=r, shape=mask_shape, device=pipe.device
        )
    prompt_list = []
    image_names = []
    subset_idx = getattr(args, "subset_idx", 0)
    num_subsets = max(1, getattr(args, "num_subsets", 1))
    for global_idx, prompt in enumerate(prompts):
        # Respect global num_prompts limit
        if args.num_prompts is not None and global_idx >= args.num_prompts:
            break
        # Shard prompts across subsets for distributed inference
        if (global_idx % num_subsets) != subset_idx:
            continue

        prompt_idx = global_idx  # keep image ids consistent with single-GPU run
        single_prompt_generation_func(
            pipe, args, output_dir, prompt, prompt_idx,
            token_upsampler, token_downsampler,
            foveation_mask=default_mask,
            full_res_foveation_mask=full_res_mask,
        )
        prompt_list.append(prompt)
        image_names.append(f"img_{prompt_idx:010d}.png")
    # write metadata.csv: full image filename and prompt per row
    metadata = pd.DataFrame({"image": image_names, "prompt": prompt_list})
    metadata.to_csv(os.path.join(output_dir, f"metadata_{subset_idx:05d}.csv"), index=False)


def run_user_study(pipe, args, output_dir, token_upsampler, token_downsampler):
    """
    Sample prompts from the prompt file; for each prompt, randomly sample a foveation mask,
    then call all three single-prompt generation functions (high_res, naive_mixed_res, ours)
    with the same seed and prompt. Only 'ours' uses LoRA; baseline and naive use base DiT.
    Runs high_res and naive for all prompts first (no LoRA), then loads LoRA once and runs
    ours for all prompts. Saves three images, upsampled mask, and fixation point per prompt.
    """
    import imageio
    assert args.prompt_dataset_path, "prompt_dataset_path required for user_study"
    num_prompts = getattr(args, "num_prompts", None) or 5
    all_prompts = load_prompt_dataset(args.prompt_dataset_path)
    k = min(num_prompts, len(all_prompts))
    if num_prompts == len(all_prompts):
        prompts = all_prompts
    else:
        prompts = random.sample(all_prompts, k)
    height, width = args.height, args.width
    device = pipe.device
    print(f"Running user study with {len(prompts)} prompts, same seed={args.seed} (LoRA only for 'ours')")

    # Reuse outline args and behavior from foveation_trajectory_grid
    outline_width_frac = getattr(args, "outline_width_frac", 0.01)
    outline_color = getattr(args, "outline_color", (255, 0, 0))
    if isinstance(outline_color, str):
        outline_color = tuple(int(x) for x in outline_color.split(","))

    pipe.clear_lora(verbose=0)
    # Collect (subdir, prompt, foveation_mask, full_res_foveation_mask, center) for second pass
    items = []

    for idx, prompt in enumerate(prompts):
        subdir = os.path.join(output_dir, f"{idx:05d}")
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, "prompt.txt"), "w") as f:
            f.write(prompt)
        center = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))
        r = 0.33 # one third of the image size (fovea 5% v.s. blend 15%)
        shape = "circular"
        foveation_mask = create_foveation_mask(height, width, center=center, r=r, shape=shape, device=device)
        full_res_foveation_mask = create_foveation_mask_full_res(height, width, center=center, r=r, shape=shape, device=device)
        items.append((subdir, prompt, foveation_mask, full_res_foveation_mask, center))

        run_high_res(pipe, args, subdir, prompt, 0, token_upsampler, token_downsampler, foveation_mask=None)
        os.rename(
            os.path.join(subdir, "img_0000000000.png"),
            os.path.join(subdir, "img_high_res.png"),
        )

        # Naive mixed-resolution baseline with optional foveation outline overlay
        run_naive_mixed_res(
            pipe,
            args,
            subdir,
            prompt,
            0,
            token_upsampler,
            token_downsampler,
            foveation_mask=foveation_mask,
            full_res_foveation_mask=full_res_foveation_mask,
        )
        os.rename(
            os.path.join(subdir, "img_0000000000.png"),
            os.path.join(subdir, "img_naive.png"),
        )
        if getattr(args, "foveation_outline", True):
            naive_path = os.path.join(subdir, "img_naive.png")
            naive_img = imageio.imread(naive_path)
            if naive_img.ndim == 2:
                naive_img = np.stack([naive_img] * 3, axis=-1)
            draw_foveation_outline(
                naive_img,
                full_res_foveation_mask,
                height,
                width,
                outline_width_frac=outline_width_frac,
                color=outline_color,
            )
            imageio.imwrite(naive_path, naive_img)

        # Save full-res mask (optionally blurred to match pipeline merge)
        mask_to_save = full_res_foveation_mask.unsqueeze(0).unsqueeze(0).float().to(device)
        if getattr(args, "soft_foveation_blend", False):
            latent_h = height // 16
            upscale = height / latent_h
            mask_to_save = _gaussian_blur_mask_2d(mask_to_save, upscale, device=device, dtype=mask_to_save.dtype)
        mask_uint8 = (np.clip(mask_to_save.squeeze().cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(subdir, "mask.png"), mask_uint8)

        cx_px = (center[0] + 0.5) * width
        cy_px = (center[1] + 0.5) * height
        dot_radius = max(2, int(0.01 * min(width, height)))
        y = np.arange(height, dtype=np.float64)[:, None]
        x = np.arange(width, dtype=np.float64)[None, :]
        dist_sq = (x - cx_px) ** 2 + (y - cy_px) ** 2
        in_circle = dist_sq <= (dot_radius ** 2)
        fixation_img = np.zeros((height, width, 3), dtype=np.uint8)
        fixation_img[in_circle] = [255, 0, 0]
        imageio.imwrite(os.path.join(subdir, "fixation_point.png"), fixation_img)
        print(f"Saved {idx:05d}/ (img_high_res.png, img_naive.png, mask.png, fixation_point.png)")

    if args.lora_checkpoint is not None:
        pipe.load_lora(pipe.dit, args.lora_checkpoint)
        print(f"Loaded LoRA checkpoint from {args.lora_checkpoint}")
    if args.dit_checkpoint is not None:
        state_dict = load_state_dict(args.dit_checkpoint, torch_dtype=torch.bfloat16)
        pipe.dit.load_state_dict(state_dict)
        print(f"Loaded DiT checkpoint from {args.dit_checkpoint}")
        
    for idx, (subdir, prompt, foveation_mask, full_res_foveation_mask, _) in enumerate(items):
        run_ours(
            pipe,
            args,
            subdir,
            prompt,
            0,
            token_upsampler,
            token_downsampler,
            foveation_mask=foveation_mask,
            full_res_foveation_mask=full_res_foveation_mask,
        )
        os.rename(
            os.path.join(subdir, "img_0000000000.png"),
            os.path.join(subdir, "img_ours.png"),
        )
        if getattr(args, "foveation_outline", True):
            ours_path = os.path.join(subdir, "img_ours.png")
            ours_img = imageio.imread(ours_path)
            if ours_img.ndim == 2:
                ours_img = np.stack([ours_img] * 3, axis=-1)
            draw_foveation_outline(
                ours_img,
                full_res_foveation_mask,
                height,
                width,
                outline_width_frac=outline_width_frac,
                color=outline_color,
            )
            imageio.imwrite(ours_path, ours_img)
        print(f"Saved {idx:05d}/ img_ours.png")
    if args.lora_checkpoint is not None:
        pipe.clear_lora(verbose=0)
    print(f"Done. User study under {output_dir}")


# additional visualizations (different foveation trajectories, runtime comparisons, etc)
def run_circular_traj(pipe, args, output_dir, token_upsampler, token_downsampler):
    """Run circular foveation mask experiment."""
    print(f"Running circular experiment with {args.num_frames} frames...")
    
    num_steps = args.num_frames
    orbit_radius = args.orbit_radius
    mask_radius = args.mask_radius
    
    print(f"Orbit Radius: {orbit_radius}, Mask Radius: {mask_radius}")
    
    for i in range(num_steps):
        # Calculate current angle
        angle = 2 * math.pi * (i / num_steps)
        
        # Compute center position (Polar -> Cartesian)
        center_x = orbit_radius * math.cos(angle)
        center_y = orbit_radius * math.sin(angle)
        center = (center_x, center_y)
        
        foveation_mask = create_foveation_mask(
            args.height, args.width, center=center, r=mask_radius, shape=args.mask_shape, device=pipe.device
        )
        full_res_foveation_mask = create_foveation_mask_full_res(
            args.height, args.width, center=center, r=mask_radius, shape=args.mask_shape, device=pipe.device
        )
        # Save full-res mask (optionally blurred to match pipeline merge)
        mask_to_save = full_res_foveation_mask.unsqueeze(0).unsqueeze(0).float().to(pipe.device)
        if getattr(args, "soft_foveation_blend", False):
            upscale = args.height / (args.height // 16)
            mask_to_save = _gaussian_blur_mask_2d(mask_to_save, upscale, pipe.device, mask_to_save.dtype)
        save_image(mask_to_save, os.path.join(output_dir, f"foveation_mask_{i:03d}.png"))
        
        # Run diffusion pipeline
        torch.cuda.empty_cache()
        start_time = time.time()
        
        vis_kwargs = {}
        foveation_mask_to_use = foveation_mask
        if getattr(args, "save_vis_vlow_x0", False):
            vis_kwargs["vis_vlow_x0_folder"] = os.path.join(output_dir, "vis_vlow_x0")
            if getattr(args, "vis_vlow_x0_steps", None):
                vis_kwargs["vis_vlow_x0_steps"] = [int(x) for x in args.vis_vlow_x0_steps.split(",")]
            foveation_mask_to_use = torch.zeros(args.height // 16, args.width // 16, device=pipe.device, dtype=torch.float32)
        pipe_kwargs = dict(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            seed=args.seed,
            rand_device="cuda",
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.guidance_scale,
            foveation_mask=foveation_mask_to_use,
            token_upsampler=token_upsampler,
            token_downsampler=token_downsampler,
            decode_mode=args.decode_mode,
            prediction_type=args.prediction_type,
            tokenAE_training=args.tokenAE_training,
            soft_foveation_blend=getattr(args, "soft_foveation_blend", False),
            **vis_kwargs,
        )
        if foveation_mask_to_use is foveation_mask:
            pipe_kwargs["full_res_foveation_mask"] = full_res_foveation_mask
        image = pipe(**pipe_kwargs)
        
        end_time = time.time()
        print(f"Step {i:02d}/{num_steps} | Angle: {math.degrees(angle):.1f}° | Time: {end_time - start_time:.2f}s")
        
        image.save(os.path.join(output_dir, f"img_{i:03d}.png"))


def run_vary_radius(pipe, args, output_dir, token_upsampler, token_downsampler):
    """Run radius ablation experiment."""
    print("Running radius ablation experiment...")
    
    r_list = np.linspace(0.1, 1.0, args.num_frames)
    
    for r_idx, r in enumerate(r_list):
        r_idx_str = str(r_idx).zfill(2)
        
        foveation_mask = create_foveation_mask(
            args.height, args.width, center=(0, 0), r=r, shape=args.mask_shape, device=pipe.device
        )
        full_res_foveation_mask = create_foveation_mask_full_res(
            args.height, args.width, center=(0, 0), r=r, shape=args.mask_shape, device=pipe.device
        )
        # Save full-res mask (optionally blurred to match pipeline merge)
        mask_to_save = full_res_foveation_mask.unsqueeze(0).unsqueeze(0).float().to(pipe.device)
        if getattr(args, "soft_foveation_blend", False):
            upscale = args.height / (args.height // 16)
            mask_to_save = _gaussian_blur_mask_2d(mask_to_save, upscale, pipe.device, mask_to_save.dtype)
        save_image(mask_to_save, os.path.join(output_dir, f"foveation_mask_{r_idx_str}.png"))
        
        # Run pipeline
        torch.cuda.empty_cache()
        start_time = time.time()
        
        vis_kwargs = {}
        foveation_mask_to_use = foveation_mask
        if getattr(args, "save_vis_vlow_x0", False):
            vis_kwargs["vis_vlow_x0_folder"] = os.path.join(output_dir, "vis_vlow_x0")
            if getattr(args, "vis_vlow_x0_steps", None):
                vis_kwargs["vis_vlow_x0_steps"] = [int(x) for x in args.vis_vlow_x0_steps.split(",")]
            foveation_mask_to_use = torch.zeros(args.height // 16, args.width // 16, device=pipe.device, dtype=torch.float32)
        pipe_kwargs = dict(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            seed=args.seed,
            rand_device="cuda",
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.guidance_scale,
            foveation_mask=foveation_mask_to_use,
            token_upsampler=token_upsampler,
            token_downsampler=token_downsampler,
            decode_mode=args.decode_mode,
            prediction_type=args.prediction_type,
            tokenAE_training=args.tokenAE_training,
            soft_foveation_blend=getattr(args, "soft_foveation_blend", False),
            **vis_kwargs,
        )
        if foveation_mask_to_use is foveation_mask:
            pipe_kwargs["full_res_foveation_mask"] = full_res_foveation_mask
        image = pipe(**pipe_kwargs)
        
        end_time = time.time()
        print(f"Radius {r:.2f} | Time: {end_time - start_time:.2f}s")
        
        image.save(os.path.join(output_dir, f"img_{r_idx_str}.png"))


# runtime experiments
def run_runtime_experiments(pipe, args, output_dir, token_upsampler, token_downsampler):
    """Run runtime experiments."""
    print("Running runtime experiments...")
    args.mask_shape = "square"
    args.num_frames = 32 # low-res latents
    # add first one again in the beginning
    #token_ratio_list = np.linspace(0.25, 1.0, args.num_frames)
    # reverse list 
    #token_ratio_list = token_ratio_list[::-1] # 1 to 0.25
    #token_ratio_list = np.concatenate([[1], token_ratio_list]) # 1 to 0.25
    # HR: r^2 tokens, LR: (1-r^2)/4 tokens -> total = (1 + 3r^2)/4 tokens = K
    # min tokens K = 0.25 (r = 0) to K = 1.0 (r = 1.0)
    # solve for r: r = sqrt((4K - 1) / 3)
    #r_list = np.sqrt((4 * token_ratio_list - 1) / 3)
    runtime_list = []
    actual_token_ratio_list = []
    #print(token_ratio_list)
    #print(r_list)
    r_list = np.arange(0, args.num_frames + 1) # 0 to 32
    r_list = r_list[::-1] # 32 to 0
    r_list = np.concatenate([[r_list[0]], r_list]) # 32, 32, ..., 0



    for r_idx, r in enumerate(r_list):
        r_idx_str = str(r_idx).zfill(2)
        
        foveation_mask = torch.zeros(args.height // 32, args.width // 32, device=pipe.device, dtype=torch.float32) # latent tokens of shape 32x32
        foveation_mask[:r, :r] = 1.0
        foveation_mask = F.interpolate(
            foveation_mask.unsqueeze(0).unsqueeze(0),
            size=(args.height // 16, args.width // 16),
            mode='nearest'
        ).squeeze(0).squeeze(0)

        
        total_orig_tokens = foveation_mask.shape[-2] * foveation_mask.shape[-1]
        num_hr_tokens = foveation_mask.sum()
        num_lr_tokens = (total_orig_tokens - num_hr_tokens) // 4
        total_tokens = num_hr_tokens + num_lr_tokens

        token_ratio = total_tokens / total_orig_tokens
        print(f"Actual token ratio: {token_ratio}, radius: {r}")
        print(f"HR tokens: {num_hr_tokens}, LR tokens: {num_lr_tokens}")
        print(f"Total tokens: {total_tokens}")
        print("--------------------------------")
        actual_token_ratio_list.append(token_ratio.item())
        # Run pipeline: average runtime over multiple runs with identical parameters
        vis_kwargs = {}

        pipe_kwargs = dict(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            seed=args.seed,
            rand_device="cuda",
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.guidance_scale,
            foveation_mask=foveation_mask,
            token_upsampler=token_upsampler,
            token_downsampler=token_downsampler,
            decode_mode=args.decode_mode,
            prediction_type=args.prediction_type,
            tokenAE_training=args.tokenAE_training,
            soft_foveation_blend=getattr(args, "soft_foveation_blend", False),
            **vis_kwargs,
        )

        num_repeats = 5
        runtimes = []
        saved_image = None
        for rep in range(num_repeats):
            torch.cuda.empty_cache()
            start_time = time.time()
            image = pipe(**pipe_kwargs)
            end_time = time.time()
            runtimes.append(end_time - start_time)
            # Save only the first image for this radius to avoid clutter
            if rep == 0:
                saved_image = image

        avg_runtime = float(np.mean(runtimes))
        print(f"Radius {r:.2f} | Avg Time over {num_repeats} runs: {avg_runtime:.2f}s")
        runtime_list.append(avg_runtime)

        if saved_image is not None:
            saved_image.save(os.path.join(output_dir, f"img_{r_idx_str}.png"))
    
    # save runtime and radius list to numpy
    runtime_list = np.array(runtime_list)
    r_list = np.array(r_list)
    token_ratio_list = np.array(actual_token_ratio_list)
    np.savez(os.path.join(output_dir, "runtime_and_radius_list.npz"), runtime_list=runtime_list, r_list=r_list, token_ratio_list=actual_token_ratio_list)


def run_foveation_trajectory_grid(pipe, args, output_dir, token_upsampler, token_downsampler):
    """
    Generate a grid of images with different foveation masks for paper figures.
    - Trajectory is fixed by args.foveation_trajectory_type: "radius" or "circular".
    - One list of foveation masks is generated once and reused for all prompts.
    - Each row = one prompt (from prompt_dataset_path), each column = one mask in the trajectory.
    - Saves to output_dir/{prompt_idx:03d}/img_{col:04d}.png with foveation outline drawn.
    """
    import imageio
    #assert args.prompt_dataset_path, "prompt_dataset_path required for foveation_trajectory_grid"

    if args.prompt_dataset_path is not None:
        all_prompts = load_prompt_dataset(args.prompt_dataset_path)
    else:
        all_prompts = [args.prompt]
        args.num_prompts = 1
        print(f"Using single prompt: {args.prompt}")

    prompt_ids = getattr(args, "prompt_ids", None)

    if prompt_ids is not None:
        if len(prompt_ids) == 0:
            raise ValueError("prompt_ids must contain at least one index when provided.")
        max_idx = len(all_prompts) - 1
        invalid = [i for i in prompt_ids if i < 0 or i > max_idx]
        if invalid:
            raise ValueError(
                f"prompt_ids contain invalid indices {invalid}; valid range is [0, {max_idx}]."
            )
        prompts = [all_prompts[i] for i in prompt_ids]
        print(
            f"Using {len(prompts)} prompts from {args.prompt_dataset_path} "
            f"with explicit prompt_ids={prompt_ids}"
        )
    else:
        assert args.num_prompts is not None and args.num_prompts > 0, "num_prompts required when prompt_ids is not provided"
        #prompts = np.random.choice(all_prompts, size=args.num_prompts, replace=False)
        prompts = all_prompts[:args.num_prompts]
        print(f"Using {len(prompts)} prompts from {args.prompt_dataset_path}")
    outline_width_frac = getattr(args, "outline_width_frac", 0.01)
    outline_color = getattr(args, "outline_color", (255, 0, 0))
    if isinstance(outline_color, str):
        outline_color = tuple(int(x) for x in outline_color.split(","))

    height, width = args.height, args.width
    device = pipe.device

    # Generate fixed foveation trajectory once (list of masks), reuse for all prompts
    foveation_masks, full_res_foveation_masks = generate_foveation_trajectory_masks(height, width, args, device)
    num_cols = len(foveation_masks)
    print(f"Trajectory type={getattr(args, 'foveation_trajectory_type', 'circular')}, {num_cols} masks")

    for prompt_idx, prompt in enumerate(prompts):
        prompt_dir = os.path.join(output_dir, f"{prompt_idx:03d}")
        os.makedirs(prompt_dir, exist_ok=True)
        with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        for col, foveation_mask in enumerate(foveation_masks):
            torch.cuda.empty_cache()
            pipe_kwargs = dict(
                prompt=prompt,
                height=height,
                width=width,
                seed=args.seed,
                rand_device="cuda",
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.guidance_scale,
                foveation_mask=foveation_mask,
                token_upsampler=token_upsampler,
                token_downsampler=token_downsampler,
                decode_mode=args.decode_mode,
                prediction_type=args.prediction_type,
                tokenAE_training=args.tokenAE_training,
                soft_foveation_blend=getattr(args, "soft_foveation_blend", False),
            )
            pipe_kwargs["full_res_foveation_mask"] = full_res_foveation_masks[col]
            image = pipe(**pipe_kwargs)
            # Optionally draw foveation outline on image
            image_np = np.array(image)
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
            if getattr(args, "foveation_outline", True):
                draw_foveation_outline(
                    image_np,
                    full_res_foveation_masks[col],
                    height,
                    width,
                    outline_width_frac=outline_width_frac,
                    color=outline_color,
                )
            out_path = os.path.join(prompt_dir, f"img_{col:04d}.png")
            imageio.imwrite(out_path, image_np)
            # Save full-res foveation mask (optionally blurred to match pipeline merge)
            mask_high = full_res_foveation_masks[col].unsqueeze(0).unsqueeze(0).float().to(device)
            if getattr(args, "soft_foveation_blend", False):
                latent_h, latent_w = height // 16, width // 16
                upscale = height / latent_h
                mask_high = _gaussian_blur_mask_2d(
                    mask_high, upscale, device=device, dtype=mask_high.dtype
                )
            mask_high = mask_high.squeeze().cpu().numpy()
            mask_high = np.clip(mask_high, 0.0, 1.0)
            mask_uint8 = (mask_high * 255).astype(np.uint8)
            mask_path = os.path.join(prompt_dir, f"mask_{col:04d}.png")
            imageio.imwrite(mask_path, mask_uint8)
            print(f"Saved {prompt_idx:03d}/img_{col:04d}.png, {prompt_idx:03d}/mask_{col:04d}.png")
    print(f"Done. Grid saved under {output_dir}")


def main(args):
    # Fix randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load pipeline
    pipe = load_pipeline(args)

    # Load LoRA checkpoint (skip for user_study: LoRA loaded only for "ours" inside run_user_study)
    if args.lora_checkpoint is not None and args.experiment != "user_study":
        pipe.load_lora(pipe.dit, args.lora_checkpoint)
        print(f"Loaded LoRA checkpoint from {args.lora_checkpoint}")
    elif args.lora_checkpoint is not None and args.experiment == "user_study":
        print("User study: LoRA will be loaded only for 'ours' runs (baseline and naive use base DiT)")
    else:
        print("No LoRA checkpoint provided, using base model")

    
    # Load DiT checkpoint 
    if args.dit_checkpoint is not None and args.experiment != "user_study":
        state_dict = load_state_dict(args.dit_checkpoint, torch_dtype=torch.bfloat16)
        pipe.dit.load_state_dict(state_dict)
        print(f"Loaded DiT checkpoint from {args.dit_checkpoint}")
    elif args.dit_checkpoint is not None and args.experiment == "user_study":
        print("User study: DiT will be loaded only for 'ours' runs (baseline and naive use base DiT)")
    else:
        print("No DiT checkpoint provided, using base model")
    
    # Load TokenAE
    if args.experiment == "high_res":
        token_upsampler, token_downsampler = None, None
        print("No TokenAE provided, using base model")
    else:
        token_upsampler, token_downsampler = load_token_ae(args, device)
        if token_upsampler is not None and token_downsampler is not None:
            count_token_ae_params(token_upsampler, token_downsampler)

    if args.full_eval:
        assert args.prompt_dataset_path is not None, "Prompt dataset path is required for full evaluation"
        prompts = load_prompt_dataset(args.prompt_dataset_path)
        print("Loaded %d prompts from %s" % (len(prompts), args.prompt_dataset_path))
    else:
        prompts = [args.prompt]
        print("Using single prompt: %s" % args.prompt)

    # Run selected experiment
    if args.experiment in SINGLE_PROMPT_GENERATION_FUNCTIONS:
        # high_res, naive_mixed_res, ours
        run_full_eval(
            SINGLE_PROMPT_GENERATION_FUNCTIONS[args.experiment],
            pipe, args, output_dir, prompts, token_upsampler, token_downsampler
        )
    elif args.experiment == "circular_traj":
        run_circular_traj(pipe, args, output_dir, token_upsampler, token_downsampler)
    elif args.experiment == "vary_radius":
        run_vary_radius(pipe, args, output_dir, token_upsampler, token_downsampler)
    elif args.experiment == "runtime":
        run_runtime_experiments(pipe, args, output_dir, token_upsampler, token_downsampler)
    elif args.experiment == "foveation_trajectory_grid":
        run_foveation_trajectory_grid(pipe, args, output_dir, token_upsampler, token_downsampler)
    elif args.experiment == "user_study":
        run_user_study(pipe, args, output_dir, token_upsampler, token_downsampler)
    else:
        print(f"Unknown experiment: {args.experiment}")
        return
    
    # Create video if applicable
    #if args.experiment in ["circular", "radius_ablation", "position_ablation"]:
    #    make_videos(output_dir, os.path.join(output_dir, "combined.mp4"), fps=20)
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX2 Foveated Image Generation (DiffSynth)")
    
    # Model configuration
    parser.add_argument('--model_id', type=str, default="black-forest-labs/FLUX.2-klein-base-4B",
                        help='HuggingFace model ID for FLUX2')
    parser.add_argument('--tokenAE_folder', default=None, type=str,
                        help='Folder containing pre-trained token_upsampler.pt, token_downsampler.pt, and tokenAE_config.yaml')
    parser.add_argument('--tokenAE_arch', type=str, default="adapter", choices=["adapter", "flux2", "dit"],
                        help='TokenAE architecture: adapter (CNN), flux2 (Flux2 token AE), dit (DiT block up/down). Overridden by tokenAE_config.yaml when loading from tokenAE_folder.')
    parser.add_argument('--lora_checkpoint', default=None, type=str, help='Path to the LoRA .safetensors checkpoint')

    # DiT checkpoint
    parser.add_argument('--dit_checkpoint', default=None, type=str, help='Path to the DiT checkpoint')
    
    # Generation parameters
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=4.0, help='Guidance scale (cfg_scale for FLUX2)')
    parser.add_argument('--prompt', type=str, 
                        default="Documentary-style imagery: a lively little dog stands still on a lush green lawn, filling the entire frame. The dog has brownish-yellow fur, with both ears perked up, and an expression that is focused and cheerful. Sunlight evenly illuminates its body, making the fur appear soft, glossy, and highly detailed. Every part of the scene is in sharp focus, from the dog's fur to the surrounding grass and distant background elements. The background shows an open grassy field with scattered wildflowers, and faint blue sky with a few white clouds, all rendered clearly and crisply. Strong realism and clarity, emphasizing texture, color, and detail throughout the image. Full-frame composition, static pose, no depth-of-field blur, everything sharp.",
                        help='Prompt for image generation')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for generation')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./outputs/flux2_foveated_diffsynth',
                        help='Directory to save generated images')
    parser.add_argument('--save_vis_vlow_x0', type=_str_to_bool, default=False,
                        help='Save v_low_unpacked and x0_low RGB visualizations to output_dir/vis_vlow_x0 at specific denoising steps.')
    parser.add_argument('--vis_vlow_x0_steps', type=str, default=None,
                        help='Comma-separated step indices to save (e.g. "0,12,25,37,49"). Default: first, 25%%, 50%%, 75%%, last.')
    parser.add_argument('--decode_mode', type=str, default="direct", choices=["direct", "merge"],
                        help='VAE decode mode: direct or merge')
    parser.add_argument('--prediction_type', type=str, default="clean", choices=["clean", "flow", "refiner"],
                        help='Prediction type for token AE: clean or flow')
    parser.add_argument('--use_as_refiner', type=_str_to_bool, default=False,
                        help='Use the token upsampler as a refiner')
    
    # Experiment configuration
    parser.add_argument('--experiment', type=str, default="circular",
                        choices=[
                            "high_res", "naive_mixed_res", "ours",
                            "circular_traj", "vary_radius",
                            "runtime", "foveation_trajectory_grid",
                            "user_study",
                        ],
                        help='Experiment to run')
    parser.add_argument('--full_eval', action='store_true', default=False,
                        help='Run full evaluation on prompt dataset')
    parser.add_argument('--subset_idx', type=int, default=0,
                        help='Subset index for distributed full_eval (0 <= subset_idx < num_subsets)')
    parser.add_argument('--num_subsets', type=int, default=1,
                        help='Total number of subsets for distributed full_eval')
    parser.add_argument('--full_eval_mask', type=str, default='square', choices=['square', 'checkerboard', 'circular'],
                        help='Foveation mask for full eval: square (center region) or checkerboard (interleaved)')
    parser.add_argument('--prompt_dataset_path', type=str, help='Path to the prompt dataset as a csv file')
    parser.add_argument('--num_prompts', type=int, default=None, help='Number of prompts to run')
    parser.add_argument('--num_cols', type=int, default=4, help='Number of columns (foveation steps per row) for foveation_trajectory_grid')
    parser.add_argument('--foveation_trajectory_type', type=str, default='circular',
                        choices=['radius', 'circular', 'polygons', 'multi_circle'],
                        help='Trajectory for foveation_trajectory_grid: radius, circular, polygons (triangle/square/hexagon/star), or multi_circle (2-3 disjoint circles per mask)')
    parser.add_argument('--outline_width_frac', type=float, default=0.005, help='Foveation outline line width as fraction of image width (default 0.01)')
    parser.add_argument('--outline_color', type=str, default='255,0,0', help='Foveation outline color as R,G,B (default red: 255,0,0)')
    parser.add_argument('--foveation_outline', type=_str_to_bool, default=False, help='Do not draw foveation border on foveation_trajectory_grid images (save plain image)')
    parser.add_argument('--soft_foveation_blend', type=_str_to_bool, default=False, help='Use soft-edged (Gaussian falloff) blend when merging low-res and high-res in decode')

    parser.add_argument(
        '--prompt_ids',
        type=int,
        nargs='+',
        default=None,
        help=(
            'Explicit prompt indices (row indices of prompt_dataset_path CSV) to use for '
            'foveation_trajectory_grid. When provided, overrides num_prompts and uses these '
            'prompts in the given order.'
        ),
    )

    
    # Circular experiment specific
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames for circular experiment')
    parser.add_argument('--orbit_radius', type=float, default=0.25, help='Distance of mask center from image center')
    parser.add_argument('--mask_radius', type=float, default=0.30, help='Radius (circular) or ratio 0-1 (square): 1=full image, 0=all black')
    parser.add_argument('--mask_shape', type=str, default='circular', choices=['circular', 'square'], help='Foveation mask shape: circular or square')

    # TokenAE configuration (used when tokenAE_config.yaml not found)
    parser.add_argument('--in_channels', type=int, default=32, help='Token AE in_channels.')
    parser.add_argument('--feature_channels', type=int, default=64, help='Token AE feature_channels.')
    parser.add_argument('--embedding_dim', type=int, default=3072, help='Token AE embedding_dim.')
    parser.add_argument('--tokenAE_dont_use_learned_components', type=_str_to_bool, default=False,
                        help='Do not use learned components for TokenAE')
    parser.add_argument('--upsampler_normalization_type', type=str, default="gn", choices=["ada", "gn", "none"], help="Normalization type for token upsampler.")
    parser.add_argument('--downsampler_normalization_type', type=str, default="none", choices=["ada", "gn", "none"], help="Normalization type for token downsampler.")
    parser.add_argument('--naive_upsample_mode', type=str, default="bilinear", choices=["nearest", "bilinear", "bicubic"], help="Naive upsample mode for token upsampler.")
    parser.add_argument('--naive_downsample_mode', type=str, default="bilinear", choices=["nearest", "bilinear", "bicubic"], help="Naive downsample mode for token downsampler.")
    parser.add_argument('--upsampler_variance_correction', default=None, type=float, help="Variance correction for token upsampler.")
    parser.add_argument('--downsampler_variance_correction', default=None, type=float, help="Variance correction for token downsampler.")
    parser.add_argument('--upsampler_num_res_blocks', type=int, default=10, help="Number of residual blocks for token upsampler.")
    parser.add_argument('--downsampler_num_res_blocks', type=int, default=2, help="Number of residual blocks for token downsampler.")

    # Flux2 TokenAE config (used when tokenAE_arch=flux2; overridden by tokenAE_config.yaml)
    parser.add_argument('--flux2_naive_downsample_mode', type=str, default=None,
                        help="Flux2 downsampler: None=learned, 'nearest'/'bilinear'=fixed interpolate.")
    parser.add_argument('--flux2_resnet_eps', type=float, default=1e-6)
    parser.add_argument('--flux2_resnet_act_fn', type=str, default="swish")
    parser.add_argument('--flux2_resnet_groups', type=int, default=32)
    parser.add_argument('--flux2_downsampler_add_norm_act_before_conv_out', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--flux2_upsampler_add_norm_act_before_conv_out', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--flux2_upsampler_num_pre_upsampler_res_blocks', type=int, default=0)
    parser.add_argument('--flux2_upsampler_naive_upsample_mode', type=str, default=None, choices=["nearest", "bilinear"],
                        help="Add residual path: output = naive_upsample(input) + network_output.")
    parser.add_argument('--flux2_downsampler_add_attention', type=_str_to_bool, default=False)
    parser.add_argument('--flux2_downsampler_attention_head_dim', type=int, default=None)
    parser.add_argument('--flux2_downsampler_attn_groups', type=int, default=None)
    parser.add_argument('--flux2_upsampler_add_attention', type=_str_to_bool, default=False)
    parser.add_argument('--flux2_upsampler_attention_head_dim', type=int, default=None)
    parser.add_argument('--flux2_upsampler_attn_groups', type=int, default=None)

    # DiT token up/down sampler (tokenAE_arch=dit; overridden by tokenAE_config.yaml)
    parser.add_argument('--upsampler_num_dit_blocks', type=int, default=1, help='Number of DiT blocks in DiT token upsampler.')
    parser.add_argument('--downsampler_num_dit_blocks', type=int, default=1, help='Number of DiT blocks in DiT token downsampler.')
    parser.add_argument('--upsampler_dit_use_temb', type=_str_to_bool, default=False,
                        help='Use time conditioning in DiT token upsampler (true/false).')
    parser.add_argument('--downsampler_dit_use_temb', type=_str_to_bool, default=False,
                        help='Use time conditioning in DiT token downsampler (true/false).')
    
    # Eval configuration
    parser.add_argument('--tokenAE_training', type=_str_to_bool, default=False, help='Train token AE (true/false).')

    args = parser.parse_args()
    main(args)