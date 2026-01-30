"""
Z-Image Foveated Diffusion Inference Script

This script demonstrates foveated image generation using the Z-Image model
with circular foveation masks that orbit around the image center.

The foveation mask defines high-resolution (HR) and low-resolution (LR) regions:
- HR regions (mask=1): Full 4 tokens per 2x2 block
- LR regions (mask=0): Single token per 2x2 block (top-left)
"""

import os
import sys
import math
import time
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Add DiffSynth-Studio to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

#from diffsynth.pipelines.z_image import ZImagePipeline
from diffsynth.pipelines.z_image_fov import ZImageFoveatedPipeline
from diffsynth.core import ModelConfig


# ============================================================================
# Simple Token Up/Downsampler (Nearest-Neighbor)
# ============================================================================

class NearestNeighborTokenDownsampler(nn.Module):
    """
    Simple 2x2 downsampler using top-left (nearest-neighbor) sampling.
    
    Takes [B, C, H, W] latents and returns [B, C, H//2, W//2] by selecting
    the top-left pixel of each 2x2 block.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x, temb=None, foveation_mask=None):
        """
        Downsample by factor of 2 using nearest-neighbor (top-left selection).
        
        Args:
            x: [B, C, H, W] input tensor
            temb: Unused (for API compatibility)
            foveation_mask: Unused (for API compatibility)
            
        Returns:
            [B, C, H//2, W//2] downsampled tensor
        """
        # Nearest-neighbor downsampling (top-left selection)
        return F.interpolate(x, scale_factor=0.5, mode='nearest')


class NearestNeighborTokenUpsampler(nn.Module):
    """
    Simple 2x2 upsampler using nearest-neighbor (pixel broadcast).
    
    Takes [B, C, H, W] latents and returns [B, C, H*2, W*2] by broadcasting
    each pixel to a 2x2 block.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x, temb=None, foveation_mask=None):
        """
        Upsample by factor of 2 using nearest-neighbor (pixel broadcast).
        
        Args:
            x: [B, C, H, W] input tensor
            temb: Unused (for API compatibility)
            foveation_mask: Unused (for API compatibility)
            
        Returns:
            [B, C, H*2, W*2] upsampled tensor
        """
        # Nearest-neighbor upsampling (pixel broadcast to 2x2 block)
        return F.interpolate(x, scale_factor=2, mode='nearest')


# ============================================================================
# Foveation Mask Creation
# ============================================================================

def create_foveation_mask(
    height: int,
    width: int,
    center: tuple = (0.0, 0.0),
    radius: float = 0.25,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Create a circular foveation mask.
    
    This follows the same logic as utils.py:
    - Center is in normalized coordinates (-0.5 to 0.5)
    - Radius is relative to half the diagonal of the image
    
    Args:
        height: Height of the mask (in latent space)
        width: Width of the mask (in latent space)
        center: (x, y) center of the circle in normalized coordinates [-0.5, 0.5]
        radius: Radius relative to half the diagonal (0.0 to 1.0+)
                - 0.0 = no mask
                - ~0.7 = covers full image for square aspect ratio
                - 1.0 = radius equals half the diagonal
        device: Target device
        
    Returns:
        Tensor [H, W] with 1.0 for high-res regions, 0.0 for low-res regions
    """
    # 1. Convert normalized center to pixel coordinates
    # center is in range (-0.5, 0.5), convert to (0, w) and (0, h)
    cx = (center[0] + 0.5) * width
    cy = (center[1] + 0.5) * height
    
    # 2. Convert normalized radius to pixel radius
    # Radius is relative to half the diagonal (consistent across aspect ratios)
    diagonal = (height**2 + width**2)**0.5
    radius_px = radius * (diagonal / 2.0)
    
    # 3. Create coordinate grid
    y_range = torch.arange(height, device=device, dtype=torch.float32)
    x_range = torch.arange(width, device=device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    
    # 4. Calculate squared Euclidean distance from center
    dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
    
    # 5. Create binary mask (1 inside circle, 0 outside)
    mask = (dist_sq <= radius_px**2).float()
    
    return mask


def save_mask_visualization(mask: torch.Tensor, path: str):
    """Save foveation mask as an image for visualization."""
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_np).save(path)


# ============================================================================
# Video Creation Utility
# ============================================================================

def make_video_from_images(image_folder: str, output_path: str, fps: int = 20):
    """Create a video from PNG images in a folder."""
    try:
        import cv2
        
        images = sorted([f for f in os.listdir(image_folder) if f.startswith("img_") and f.endswith(".png")])
        if not images:
            print("No images found for video creation")
            return
        
        # Read first image to get dimensions
        first_img = cv2.imread(os.path.join(image_folder, images[0]))
        height, width = first_img.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for img_name in images:
            img_path = os.path.join(image_folder, img_name)
            img = cv2.imread(img_path)
            video.write(img)
        
        video.release()
        print(f"Video saved to {output_path}")
    except ImportError:
        print("OpenCV not installed, skipping video creation")
    except Exception as e:
        print(f"Error creating video: {e}")


# ============================================================================
# Main Inference Function
# ============================================================================

def main(args):
    # Fix randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========================================================================
    # Load Z-Image Pipeline
    # ========================================================================
    print("Loading Z-Image pipeline...")
    
    # Model configs (shared between standard and foveated pipelines)
    model_configs = [
        ModelConfig(path=[
            os.path.join(args.model_folder, "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"),
            os.path.join(args.model_folder, "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"),
        ]),
        ModelConfig(path=[
            os.path.join(args.model_folder, "text_encoder/model-00001-of-00003.safetensors"),
            os.path.join(args.model_folder, "text_encoder/model-00002-of-00003.safetensors"),
            os.path.join(args.model_folder, "text_encoder/model-00003-of-00003.safetensors"),
        ]),
        ModelConfig(path=os.path.join(args.model_folder, "vae/diffusion_pytorch_model.safetensors")),
    ]
    tokenizer_config = ModelConfig(path=os.path.join(args.model_folder, "tokenizer/"))
    
    # Use foveated pipeline for CRPA attention support
    pipe = ZImageFoveatedPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    # ========================================================================
    # Initialize Token Up/Downsampler (Nearest-Neighbor)
    # ========================================================================
    print("Initializing nearest-neighbor token up/downsampler...")
    
    token_upsampler = NearestNeighborTokenUpsampler().to(device).to(torch.bfloat16)
    token_downsampler = NearestNeighborTokenDownsampler().to(device).to(torch.bfloat16)

    # ========================================================================
    # Run Experiment
    # ========================================================================
    prompt = args.prompt

    if args.experiment == "circular":
        run_circular_experiment(
            pipe, prompt, args,
            token_upsampler, token_downsampler,
            device
        )
    elif args.experiment == "radius_sweep":
        run_radius_sweep_experiment(
            pipe, prompt, args,
            token_upsampler, token_downsampler,
            device
        )
    elif args.experiment == "baseline":
        run_baseline_experiment(pipe, prompt, args, device)
    else:
        print(f"Unknown experiment: {args.experiment}")
        return

    # Create video from images
    if args.experiment in ["circular", "radius_sweep"]:
        make_video_from_images(args.output_dir, os.path.join(args.output_dir, "combined.mp4"), fps=20)


def run_foveation_experiment(
    pipe,
    prompt: str,
    args,
    token_upsampler: nn.Module,
    token_downsampler: nn.Module,
    device: str,
    centers: list,
    radii: list,
    experiment_name: str = "experiment"
):
    """
    Run foveation experiment with specified centers and radii.
    
    This is the common experiment runner that takes lists of centers and radii,
    allowing different experiment types to reuse the same pipeline code.
    
    Args:
        pipe: The diffusion pipeline
        prompt: Text prompt for generation
        args: Command line arguments
        token_upsampler: Token upsampling module
        token_downsampler: Token downsampling module
        device: Target device
        centers: List of (x, y) center coordinates for each frame
        radii: List of mask radii for each frame
        experiment_name: Name for logging
    """
    num_steps = len(centers)
    assert len(radii) == num_steps, "centers and radii must have the same length"
    
    # Calculate latent dimensions
    # Z-Image uses VAE with 8x downsampling (1024 --> 128 --> 64 since packed)
    latent_height = args.height // 8 // 2
    latent_width = args.width // 8 // 2
    
    # For foveation, we work with half the latent resolution (2x2 blocks) (32, 32)
    fov_height = latent_height // 2
    fov_width = latent_width // 2

    print(f"Running {experiment_name}: {num_steps} frames")
    print(f"  Latent size: {latent_height}x{latent_width}")
    print(f"  Foveation mask size: {fov_height}x{fov_width}")
    print(f"  Use foveation: {args.use_foveation}")

    for i in range(num_steps):
        center = centers[i]
        radius = radii[i]
        
        # Create the foveation mask at low resolution (block-level)
        foveation_mask_low = create_foveation_mask(
            fov_height, fov_width,
            center=center,
            radius=radius,
            device=device
        )
        
        # Save low-res mask visualization
        save_mask_visualization(
            foveation_mask_low,
            os.path.join(args.output_dir, f"foveation_mask_low_res_{i:03d}.png")
        )
        
        # Upsample mask to full latent resolution (for compatibility)
        foveation_mask_high = F.interpolate(
            foveation_mask_low.unsqueeze(0).unsqueeze(0),
            size=(latent_height, latent_width),
            mode='nearest'
        ).squeeze()
        
        # Save high-res mask visualization
        save_mask_visualization(
            foveation_mask_high,
            os.path.join(args.output_dir, f"foveation_mask_high_res_{i:03d}.png")
        )
        
        # Run diffusion pipeline
        torch.cuda.empty_cache()
        start_time = time.time()
        
        if args.use_foveation:
            image = pipe(
                prompt=prompt,
                seed=args.seed,
                rand_device="cuda",
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                height=args.height,
                width=args.width,
                foveation_mask=foveation_mask_high,
                foveation_start_step=args.foveation_start_step,
                token_upsampler=token_upsampler,
                token_downsampler=token_downsampler,
                vae_decode_mode=args.vae_decode_mode,
                vae_merge_downsample_mode=args.vae_merge_downsample_mode,
            )
        else:
            image = pipe(
                prompt=prompt,
                seed=args.seed,
                rand_device="cuda",
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                height=args.height,
                width=args.width,
            )
        
        end_time = time.time()
        print(f"Step {i:02d}/{num_steps} | Center: ({center[0]:.2f}, {center[1]:.2f}) | Radius: {radius:.2f} | Time: {end_time - start_time:.2f}s")
        
        # Save output image
        image.save(os.path.join(args.output_dir, f"img_{i:03d}.png"))


def run_circular_experiment(
    pipe,
    prompt: str,
    args,
    token_upsampler: nn.Module,
    token_downsampler: nn.Module,
    device: str
):
    """
    Run circular foveation mask experiment.
    
    The foveation mask orbits around the image center, creating a sequence
    of images where the high-resolution region follows a circular path.
    """
    num_frames = args.num_frames
    orbit_radius = args.orbit_radius
    mask_radius = args.mask_radius
    
    print(f"Circular experiment: orbit_radius={orbit_radius}, mask_radius={mask_radius}")
    
    # Generate centers (orbiting) and radii (constant)
    centers = []
    radii = []
    for i in range(num_frames):
        angle = 2 * math.pi * (i / num_frames)
        center_x = orbit_radius * math.cos(angle)
        center_y = orbit_radius * math.sin(angle)
        centers.append((center_x, center_y))
        radii.append(mask_radius)
    
    run_foveation_experiment(
        pipe, prompt, args,
        token_upsampler, token_downsampler, device,
        centers=centers,
        radii=radii,
        experiment_name="circular"
    )


def run_radius_sweep_experiment(
    pipe,
    prompt: str,
    args,
    token_upsampler: nn.Module,
    token_downsampler: nn.Module,
    device: str
):
    """
    Run radius sweep experiment.
    
    The foveation mask stays centered at (0, 0) while the radius grows
    from radius_start to radius_end over num_frames.
    """
    num_frames = args.num_frames
    radius_start = args.radius_start
    radius_end = args.radius_end
    
    print(f"Radius sweep experiment: center=(0, 0), radius={radius_start} -> {radius_end}")
    
    # Generate centers (fixed at origin) and radii (sweeping)
    centers = []
    radii = []
    for i in range(num_frames):
        centers.append((0.0, 0.0))
        # Linear interpolation from radius_start to radius_end
        if num_frames > 1:
            t = i / (num_frames - 1)
        else:
            t = 0.0
        radius = radius_start + t * (radius_end - radius_start)
        radii.append(radius)
    
    run_foveation_experiment(
        pipe, prompt, args,
        token_upsampler, token_downsampler, device,
        centers=centers,
        radii=radii,
        experiment_name="radius_sweep"
    )


def run_baseline_experiment(pipe, prompt: str, args, device: str):
    """Run baseline experiment without foveation."""
    print("Running baseline experiment (no foveation)...")
    
    image = pipe(
        prompt=prompt,
        seed=args.seed,
        rand_device="cuda",
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        height=args.height,
        width=args.width,
    )
    
    image.save(os.path.join(args.output_dir, "img_baseline.png"))
    print(f"Saved baseline image to {args.output_dir}/img_baseline.png")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z-Image Foveated Diffusion Inference")
    
    # Model paths
    parser.add_argument('--model_folder', type=str, 
                        default="/miele/brian/modelscope/Z-Image",
                        help='Path to Z-Image model folder')
    
    # Image parameters
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str,
                        default="Documentary-style imagery: a lively little dog stands still on a lush green lawn, filling the entire frame. The dog has brownish-yellow fur, with both ears perked up, and an expression that is focused and cheerful. Sunlight evenly illuminates its body, making the fur appear soft, glossy, and highly detailed. Every part of the scene is in sharp focus, from the dog's fur to the surrounding grass and distant background elements. The background shows an open grassy field with scattered wildflowers, and faint blue sky with a few white clouds, all rendered clearly and crisply. Strong realism and clarity, emphasizing texture, color, and detail throughout the image. Full-frame composition, static pose, no depth-of-field blur, everything sharp.",
                        help='Prompt for image generation')
    parser.add_argument('--cfg_scale', type=float, default=4.0, help='CFG guidance scale')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Foveation parameters
    # Note: Radius values are relative to half the diagonal of the image
    # For a square image: radius ~0.7 covers the full image, radius 1.0 = half diagonal
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames for experiment')
    parser.add_argument('--orbit_radius', type=float, default=0.15, 
                        help='Distance of mask center from image center, relative to half diagonal (0.0=center, 0.7=edge for square) [circular]')
    parser.add_argument('--mask_radius', type=float, default=0.25,
                        help='Radius of the foveation mask, relative to half diagonal (~0.7 covers full square image) [circular]')
    parser.add_argument('--radius_start', type=float, default=0.05,
                        help='Starting radius for radius sweep, relative to half diagonal [radius_sweep]')
    parser.add_argument('--radius_end', type=float, default=0.8,
                        help='Ending radius for radius sweep, relative to half diagonal (~0.7 covers full square) [radius_sweep]')
    parser.add_argument('--foveation_start_step', type=int, default=0,
                        help='Step to start applying foveation')
    parser.add_argument('--use_foveation', action='store_true',
                        help='Enable foveation (requires ZImageFoveatedPipeline)')
    parser.add_argument('--vae_decode_mode', type=str, default='direct',
                        choices=['direct', 'merge'],
                        help='VAE decode mode: "direct" (decode full latents) or "merge" (decode LR+HR separately, then merge)')
    parser.add_argument('--vae_merge_downsample_mode', type=str, default='average',
                        choices=['average', 'nearest'],
                        help='Downsample mode for merge VAE decode: "average" (2x2 avg pool) or "nearest" (top-left sampling)')
    
    # Experiment selection
    parser.add_argument('--experiment', type=str, default="circular",
                        choices=["circular", "radius_sweep", "baseline"],
                        help='Experiment type: circular (orbiting mask), radius_sweep (growing mask from center), baseline (no foveation)')
    
    # Output
    parser.add_argument('--output_dir', type=str, 
                        default='./outputs/z_image_foveated',
                        help='Output directory')
    
    args = parser.parse_args()
    main(args)
