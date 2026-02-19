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

# single-prompt generation
def run_high_res(pipe, args, output_dir, prompt, prompt_idx, token_upsampler=None, token_downsampler=None):
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
    )
    image.save(os.path.join(output_dir, f"img_{prompt_idx:010d}.png"))

def run_naive_mixed_res(pipe, args, output_dir, prompt, prompt_idx, token_upsampler, token_downsampler):
    """Run naive mixed-resolution generation."""
    # TODO
    pass

def run_ours(pipe, args, output_dir, prompt, prompt_idx, token_upsampler, token_downsampler):
    """Run our proposed method."""
    # TODO
    pass

SINGLE_PROMPT_GENERATION_FUNCTIONS = {
    "high_res": run_high_res,
    "naive_mixed_res": run_naive_mixed_res,
    "ours": run_ours,
}

# full-prompt generation wrapper. calls the single-prompt generation function for each prompt.
def run_full_eval(single_prompt_generation_func, pipe, args, output_dir, prompts, token_upsampler=None, token_downsampler=None):
    """Run full-prompt generation."""
    # TODO: parallelize over GPUs
    for prompt_idx, prompt in enumerate(prompts):
        single_prompt_generation_func(
            pipe, args, output_dir, prompt, prompt_idx,
            token_upsampler, token_downsampler
        )

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
        
        vis_kwargs = {}
        foveation_mask_to_use = foveation_mask_high.squeeze().to(pipe.device)
        if getattr(args, "save_vis_vlow_x0", False):
            vis_kwargs["vis_vlow_x0_folder"] = os.path.join(output_dir, "vis_vlow_x0")
            if getattr(args, "vis_vlow_x0_steps", None):
                vis_kwargs["vis_vlow_x0_steps"] = [int(x) for x in args.vis_vlow_x0_steps.split(",")]
            foveation_mask_to_use = torch.zeros(args.height // 16, args.width // 16, device=pipe.device, dtype=torch.float32)
        image = pipe(
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
            **vis_kwargs,
        )
        
        end_time = time.time()
        print(f"Step {i:02d}/{num_steps} | Angle: {math.degrees(angle):.1f}° | Time: {end_time - start_time:.2f}s")
        
        image.save(os.path.join(output_dir, f"img_{i:03d}.png"))


def run_vary_radius(pipe, args, output_dir, token_upsampler, token_downsampler):
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
        
        vis_kwargs = {}
        foveation_mask_to_use = foveation_mask_high.squeeze().to(pipe.device)
        if getattr(args, "save_vis_vlow_x0", False):
            vis_kwargs["vis_vlow_x0_folder"] = os.path.join(output_dir, "vis_vlow_x0")
            if getattr(args, "vis_vlow_x0_steps", None):
                vis_kwargs["vis_vlow_x0_steps"] = [int(x) for x in args.vis_vlow_x0_steps.split(",")]
            foveation_mask_to_use = torch.zeros(args.height // 16, args.width // 16, device=pipe.device, dtype=torch.float32)
        image = pipe(
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
            **vis_kwargs,
        )
        
        end_time = time.time()
        print(f"Radius {r:.2f} | Time: {end_time - start_time:.2f}s")
        
        image.save(os.path.join(output_dir, f"img_{r_idx_str}.png"))

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

    # Load LoRA checkpoint
    if args.lora_checkpoint is not None:
        pipe.load_lora(pipe.dit,args.lora_checkpoint)
        print(f"Loaded LoRA checkpoint from {args.lora_checkpoint}")
    else:
        print("No LoRA checkpoint provided, using base model")
    
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
    parser.add_argument('--save_vis_vlow_x0', action='store_true',
                        help='Save v_low_unpacked and x0_low RGB visualizations to output_dir/vis_vlow_x0 at specific denoising steps.')
    parser.add_argument('--vis_vlow_x0_steps', type=str, default=None,
                        help='Comma-separated step indices to save (e.g. "0,12,25,37,49"). Default: first, 25%%, 50%%, 75%%, last.')
    parser.add_argument('--decode_mode', type=str, default="direct", choices=["direct", "merge"],
                        help='VAE decode mode: direct or merge')
    parser.add_argument('--prediction_type', type=str, default="clean", choices=["clean", "flow", "refiner"],
                        help='Prediction type for token AE: clean or flow')
    parser.add_argument('--use_as_refiner', action='store_true', default=False,
                        help='Use the token upsampler as a refiner')
    
    # Experiment configuration
    parser.add_argument('--experiment', type=str, default="circular",
                        choices=[
                            "high_res", "naive_mixed_res", "ours",
                            "circular", "radius_ablation", 
                        ],
                        help='Experiment to run')
    parser.add_argument('--full_eval', action='store_true', default=False,
                        help='Run full evaluation on prompt dataset')
    parser.add_argument('--prompt_dataset_path', type=str, help='Path to the prompt dataset as a csv file')

    
    # Circular experiment specific
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames for circular experiment')
    parser.add_argument('--orbit_radius', type=float, default=0.25, help='Distance of mask center from image center')
    parser.add_argument('--mask_radius', type=float, default=0.30, help='Radius of the foveation mask')

    # TokenAE configuration (used when tokenAE_config.yaml not found)
    parser.add_argument('--in_channels', type=int, default=32, help='Token AE in_channels.')
    parser.add_argument('--feature_channels', type=int, default=64, help='Token AE feature_channels.')
    parser.add_argument('--embedding_dim', type=int, default=3072, help='Token AE embedding_dim.')
    parser.add_argument('--tokenAE_dont_use_learned_components', action='store_true', default=False,
                        help='Do not use learned components for TokenAE')
    parser.add_argument('--upsampler_normalization_type', type=str, default="gn", choices=["ada", "gn", "none"], help="Normalization type for token upsampler.")
    parser.add_argument('--downsampler_normalization_type', type=str, default="none", choices=["ada", "gn", "none"], help="Normalization type for token downsampler.")
    parser.add_argument('--naive_upsample_mode', type=str, default="nearest", choices=["nearest", "bilinear", "bicubic"], help="Naive upsample mode for token upsampler.")
    parser.add_argument('--naive_downsample_mode', type=str, default="nearest", choices=["nearest", "bilinear", "bicubic"], help="Naive downsample mode for token downsampler.")
    parser.add_argument('--upsampler_variance_correction', default=None, type=float, help="Variance correction for token upsampler.")
    parser.add_argument('--downsampler_variance_correction', default=None, type=float, help="Variance correction for token downsampler.")
    parser.add_argument('--upsampler_num_res_blocks', type=int, default=10, help="Number of residual blocks for token upsampler.")
    parser.add_argument('--downsampler_num_res_blocks', type=int, default=10, help="Number of residual blocks for token downsampler.")

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
    parser.add_argument('--flux2_downsampler_add_attention', action='store_true', default=False)
    parser.add_argument('--flux2_downsampler_attention_head_dim', type=int, default=None)
    parser.add_argument('--flux2_downsampler_attn_groups', type=int, default=None)
    parser.add_argument('--flux2_upsampler_add_attention', action='store_true', default=False)
    parser.add_argument('--flux2_upsampler_attention_head_dim', type=int, default=None)
    parser.add_argument('--flux2_upsampler_attn_groups', type=int, default=None)

    # DiT token up/down sampler (tokenAE_arch=dit; overridden by tokenAE_config.yaml)
    parser.add_argument('--upsampler_num_dit_blocks', type=int, default=1, help='Number of DiT blocks in DiT token upsampler.')
    parser.add_argument('--downsampler_num_dit_blocks', type=int, default=1, help='Number of DiT blocks in DiT token downsampler.')
    parser.add_argument('--upsampler_dit_use_temb', type=_str_to_bool, default=False,
                        help='Use time conditioning in DiT token upsampler (true/false).')
    parser.add_argument('--downsampler_dit_use_temb', type=_str_to_bool, default=False,
                        help='Use time conditioning in DiT token downsampler (true/false).')

    args = parser.parse_args()
    main(args)
