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

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../image_generation"))

from diffsynth.pipelines.flux2_image_foveated import Flux2FoveatedImagePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

# Import Token Upsampler/Downsampler from image_generation
try:
    from adapter_foveation_time_cond import TokenDownsampler, TokenUpsampler
except ImportError:
    print("Warning: Could not import TokenUpsampler/TokenDownsampler from adapter_foveation_time_cond")
    print("Make sure the file is in the path or provide pre-trained weights via --tokenAE_folder")
    TokenUpsampler = None
    TokenDownsampler = None


def create_foveation_mask(
    h: int, 
    w: int, 
    center: tuple = (0.0, 0.0), 
    r: float = 0.25, 
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Creates a binary circular mask using PyTorch.

    Args:
        h, w (int): Dimensions of the mask.
        center (tuple): (x, y) coordinates (-0.5 to 0.5).
        r (float): Radius relative to half the diagonal.
        device: The device to create the tensor on (cpu or cuda).

    Returns:
        torch.Tensor: Boolean mask of shape (h, w).
    """
    # 1. Convert normalized center to pixel coordinates
    cx = (center[0] + 0.5) * w
    cy = (center[1] + 0.5) * h

    # 2. Convert normalized radius to pixel radius
    diagonal = (h**2 + w**2)**0.5
    radius_px = r * (diagonal / 2.0)

    # 3. Create coordinate grid
    y_range = torch.arange(h, device=device, dtype=torch.float32)
    x_range = torch.arange(w, device=device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')

    # 4. Calculate squared Euclidean distance
    dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2

    # 5. Create binary mask
    mask = dist_sq <= radius_px**2

    return mask


def make_videos(folder, output_file, fps=5):
    """Combine generated images and masks into a video."""
    try:
        import imageio
        import cv2
    except ImportError:
        print("Warning: imageio or cv2 not installed, skipping video creation")
        return
    
    image_files = [f for f in sorted(os.listdir(folder)) if f.startswith("img") and f.endswith('.png')]
    mask_files = [f for f in sorted(os.listdir(folder)) if f.startswith("foveation_mask_high_res") and f.endswith('.png')]
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


def load_token_ae(args, device):
    """Load TokenUpsampler and TokenDownsampler."""
    if TokenUpsampler is not None and TokenDownsampler is not None:
        print("Creating new TokenAE (untrained)")
        # FLUX2 uses inner_dim = 48 * 128 = 6144
        token_upsampler = TokenUpsampler(
            in_channels=32,
            feature_channels=64, 
            embedding_dim=6144,
            num_res_blocks=args.upsampler_num_res_blocks,
            naive_upsample_mode=args.naive_upsample_mode,
            learn_residual=True,
            variance_correction=args.upsampler_variance_correction,
            normalization_type=args.upsampler_normalization_type,
            use_learned_components=not args.tokenAE_dont_use_learned_components
        )
        token_downsampler = TokenDownsampler(
            in_channels=32,
            feature_channels=64, 
            embedding_dim=6144,
            num_res_blocks=args.downsampler_num_res_blocks,
            naive_downsample_mode=args.naive_downsample_mode,
            learn_residual=True,
            variance_correction=args.downsampler_variance_correction,
            normalization_type=args.downsampler_normalization_type,
            use_learned_components=not args.tokenAE_dont_use_learned_components
        )
    else:
        print("No TokenAE class provided, using base model")
        return None, None
    
    if args.tokenAE_folder is not None:
        print(f"Loading TokenAE from {args.tokenAE_folder}")
        token_upsampler.load_state_dict(torch.load(
            os.path.join(args.tokenAE_folder, "token_upsampler.pt"), 
            map_location='cpu', 
            weights_only=False
        ))
        token_downsampler.load_state_dict(torch.load(
            os.path.join(args.tokenAE_folder, "token_downsampler.pt"), 
            map_location='cpu', 
            weights_only=False
        ))
    else:
        print("Not loading TokenAE checkpoints, using non-trained version.")
    
    token_upsampler = token_upsampler.to(device).to(torch.bfloat16)
    token_downsampler = token_downsampler.to(device).to(torch.bfloat16)
    
    return token_upsampler, token_downsampler


def run_baseline(pipe, args, output_dir):
    """Run baseline full-resolution generation."""
    print("Running baseline experiment...")
    args.prompt = "Documentary-style imagery: a lively little dog stands still on a lush green lawn, filling the entire frame. The dog has brownish-yellow fur, with both ears perked up, and an expression that is focused and cheerful. Sunlight evenly illuminates its body, making the fur appear soft, glossy, and highly detailed. Every part of the scene is in sharp focus, from the dog's fur to the surrounding grass and distant background elements. The background shows an open grassy field with scattered wildflowers, and faint blue sky with a few white clouds, all rendered clearly and crisply. Strong realism and clarity, emphasizing texture, color, and detail throughout the image. Full-frame composition, static pose, no depth-of-field blur, everything sharp."
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        rand_device="cuda",
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.guidance_scale,
        foveation_mask=None,
    )
    
    image.save(os.path.join(output_dir, "img_baseline.png"))
    print(f"Saved baseline image to {output_dir}/img_baseline.png")


def run_circular_experiment(pipe, args, output_dir, token_upsampler, token_downsampler):
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
        
        # Create mask at low resolution then upsample
        foveation_mask = create_foveation_mask(
            args.height // 16 // 2, 
            args.width // 16 // 2, 
            center=center, 
            r=mask_radius
        )
        
        # Process dimensions [1, 1, H, W]
        foveation_mask = foveation_mask.unsqueeze(0).unsqueeze(0).float()
        
        # Save debug visualizations
        save_image(foveation_mask, os.path.join(output_dir, f"foveation_mask_low_res_{i:03d}.png"))
        
        # Upsample for the pipeline
        foveation_mask_high = F.interpolate(
            foveation_mask, 
            size=(args.height // 16, args.width // 16), 
            mode='nearest'
        )
        save_image(foveation_mask_high, os.path.join(output_dir, f"foveation_mask_high_res_{i:03d}.png"))
        
        # Run diffusion pipeline
        torch.cuda.empty_cache()
        start_time = time.time()
        
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            seed=args.seed,
            rand_device="cuda",
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.guidance_scale,
            foveation_mask=foveation_mask_high.squeeze().to(pipe.device),
            token_upsampler=token_upsampler,
            token_downsampler=token_downsampler,
            decode_mode=args.decode_mode,
        )
        
        end_time = time.time()
        print(f"Step {i:02d}/{num_steps} | Angle: {math.degrees(angle):.1f}° | Time: {end_time - start_time:.2f}s")
        
        image.save(os.path.join(output_dir, f"img_{i:03d}.png"))


def run_radius_ablation(pipe, args, output_dir, token_upsampler, token_downsampler):
    """Run radius ablation experiment."""
    print("Running radius ablation experiment...")
    
    r_list = np.linspace(0.1, 1.0, args.num_frames)
    
    for r_idx, r in enumerate(r_list):
        r_idx_str = str(r_idx).zfill(2)
        
        # Create low-res foveation mask at center
        foveation_mask = create_foveation_mask(
            args.height // 16 // 2, 
            args.width // 16 // 2, 
            center=(0, 0), 
            r=r
        )
        
        # Process dimensions
        foveation_mask = foveation_mask.unsqueeze(0).unsqueeze(0).float()
        
        # Upsample
        foveation_mask_high = F.interpolate(
            foveation_mask, 
            size=(args.height // 16, args.width // 16), 
            mode='nearest'
        )
        save_image(foveation_mask_high, os.path.join(output_dir, f"foveation_mask_high_res_{r_idx_str}.png"))
        
        # Run pipeline
        torch.cuda.empty_cache()
        start_time = time.time()
        
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            seed=args.seed,
            rand_device="cuda",
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.guidance_scale,
            foveation_mask=foveation_mask_high.squeeze().to(pipe.device),
            token_upsampler=token_upsampler,
            token_downsampler=token_downsampler,
            decode_mode=args.decode_mode,
        )
        
        end_time = time.time()
        print(f"Radius {r:.2f} | Time: {end_time - start_time:.2f}s")
        
        image.save(os.path.join(output_dir, f"img_{r_idx_str}.png"))
        exit()


def run_position_ablation(pipe, args, output_dir, token_upsampler, token_downsampler):
    """Run position ablation experiment."""
    print("Running position ablation experiment...")
    
    for y_idx, center_y in enumerate(np.linspace(-0.25, 0.25, 5)):
        for x_idx, center_x in enumerate(np.linspace(-0.25, 0.25, 5)):
            center = (center_x, center_y)
            
            # Create low-res foveation mask
            foveation_mask = create_foveation_mask(
                args.height // 16 // 2, 
                args.width // 16 // 2, 
                center=center, 
                r=0.3
            )
            
            foveation_mask = foveation_mask.unsqueeze(0).unsqueeze(0).float()
            save_image(foveation_mask, os.path.join(output_dir, f"foveation_mask_low_res_{x_idx}_{y_idx}.png"))
            
            foveation_mask_high = F.interpolate(
                foveation_mask, 
                size=(args.height // 16, args.width // 16), 
                mode='nearest'
            )
            save_image(foveation_mask_high, os.path.join(output_dir, f"foveation_mask_high_res_{x_idx}_{y_idx}.png"))
            
            torch.cuda.empty_cache()
            
            image = pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                seed=args.seed,
                rand_device="cuda",
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.guidance_scale,
                foveation_mask=foveation_mask_high.squeeze().to(pipe.device),
                token_upsampler=token_upsampler,
                token_downsampler=token_downsampler,
                decode_mode=args.decode_mode,
            )
            
            image.save(os.path.join(output_dir, f"img_{x_idx}_{y_idx}.png"))
            print(f"Position ({center_x:.2f}, {center_y:.2f}) done")


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
    
    if args.lora_checkpoint is not None:
        pipe.load_lora(pipe.dit,args.lora_checkpoint)
        print(f"Loaded LoRA checkpoint from {args.lora_checkpoint}")
    else:
        print("No LoRA checkpoint provided, using base model")
    
    # Load TokenAE
    token_upsampler, token_downsampler = load_token_ae(args, device)
    
    # Run selected experiment
    if args.experiment == "baseline":
        run_baseline(pipe, args, output_dir)
    elif args.experiment == "circular":
        run_circular_experiment(pipe, args, output_dir, token_upsampler, token_downsampler)
    elif args.experiment == "radius_ablation":
        run_radius_ablation(pipe, args, output_dir, token_upsampler, token_downsampler)
    elif args.experiment == "position_ablation":
        run_position_ablation(pipe, args, output_dir, token_upsampler, token_downsampler)
    else:
        print(f"Unknown experiment: {args.experiment}")
        return
    
    # Create video if applicable
    if args.experiment in ["circular", "radius_ablation", "position_ablation"]:
        make_videos(output_dir, os.path.join(output_dir, "combined.mp4"), fps=20)
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX2 Foveated Image Generation (DiffSynth)")
    
    # Model configuration
    parser.add_argument('--model_id', type=str, default="black-forest-labs/FLUX.2-klein-base-4B",
                        help='HuggingFace model ID for FLUX2')
    parser.add_argument('--tokenAE_folder', default=None, type=str,
                        help='Folder containing pre-trained token_upsampler.pt and token_downsampler.pt')
    parser.add_argument('--lora_checkpoint', default=None, type=str, help='Path to the LoRA .safetensors checkpoint')
    
    
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
    parser.add_argument('--decode_mode', type=str, default="direct", choices=["direct", "merge"],
                        help='VAE decode mode: direct or merge')
    
    # Experiment configuration
    parser.add_argument('--experiment', type=str, default="circular",
                        choices=["baseline", "circular", "radius_ablation", "position_ablation"],
                        help='Experiment to run')
    
    # Circular experiment specific
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames for circular experiment')
    parser.add_argument('--orbit_radius', type=float, default=0.25, help='Distance of mask center from image center')
    parser.add_argument('--mask_radius', type=float, default=0.30, help='Radius of the foveation mask')

    # TokenAE configuration
    parser.add_argument('--tokenAE_dont_use_learned_components', action='store_true', default=False,
                        help='Do not use learned components for TokenAE')
    parser.add_argument('--upsampler_normalization_type', type=str, default="gn", choices=["ada", "gn", "none"], help="Normalization type for token upsampler.")
    parser.add_argument('--downsampler_normalization_type', type=str, default="none", choices=["ada", "gn", "none"], help="Normalization type for token downsampler.")
    parser.add_argument('--naive_upsample_mode', type=str, default="nearest", choices=["nearest", "bilinear"], help="Naive upsample mode for token upsampler.")
    parser.add_argument('--naive_downsample_mode', type=str, default="nearest", choices=["nearest", "bilinear"], help="Naive downsample mode for token downsampler.")
    parser.add_argument('--upsampler_variance_correction', default=None, type=float, help="Variance correction for token upsampler.")
    parser.add_argument('--downsampler_variance_correction', default=None, type=float, help="Variance correction for token downsampler.")
    parser.add_argument('--upsampler_num_res_blocks', type=int, default=10, help="Number of residual blocks for token upsampler.")
    parser.add_argument('--downsampler_num_res_blocks', type=int, default=10, help="Number of residual blocks for token downsampler.")

    args = parser.parse_args()
    main(args)
