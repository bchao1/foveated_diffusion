"""
FLUX2 Foveated Image Generation Pipeline for DiffSynth.

This pipeline integrates foveation logic from pipeline_flux2_klein_foveation.py
into the DiffSynth FLUX2 pipeline architecture.

Key features:
- Mixed-resolution latent processing for efficient foveated rendering
- X0 (clean image) prediction for better upsampling quality
- Phase-aligned RoPE for consistent position encoding across resolutions
- CRPA attention for efficient mixed-resolution attention
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")
import math
import torchvision
import numpy as np
import torch.nn.functional as F
from PIL import Image
from typing import Union, List, Optional, Tuple
from tqdm import tqdm
from einops import rearrange

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput

from transformers import AutoProcessor, AutoTokenizer
from ..models.flux2_text_encoder import Flux2TextEncoder
from ..models.flux2_dit_foveated import Flux2DiTFoveated
from ..models.flux2_vae import Flux2VAE
from ..models.z_image_text_encoder import ZImageTextEncoder


def _gaussian_blur_mask_2d(
    mask: torch.Tensor,
    sigma_pixels: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply separable Gaussian blur to a [1, 1, H, W] mask. sigma_pixels controls falloff width."""
    # Kernel size odd, ~3 sigma on each side
    k = max(3, int(math.ceil(3 * sigma_pixels)) * 2 + 1)
    if k % 2 == 0:
        k += 1
    x = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma_pixels ** 2 + 1e-6))
    g = g / g.sum()
    # [1, 1, k, 1] and [1, 1, 1, k] for separable conv
    kernel_h = g.view(1, 1, k, 1)
    kernel_w = g.view(1, 1, 1, k)
    pad = k // 2
    out = F.conv2d(mask, kernel_h, padding=(pad, 0))
    out = F.conv2d(out, kernel_w, padding=(0, pad))
    return out


class Flux2FoveatedImagePipeline(BasePipeline):
    """
    FLUX2 Foveated Image Generation Pipeline.
    
    This pipeline extends the standard FLUX2 pipeline with support for:
    - Foveated rendering with mixed-resolution latents
    - X0 prediction upsampling for better quality
    - CRPA attention for efficient attention computation
    """

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("FLUX.2")
        self.text_encoder: Flux2TextEncoder = None
        self.text_encoder_qwen3: ZImageTextEncoder = None
        self.dit: Flux2DiTFoveated = None
        self.vae: Flux2VAE = None
        self.tokenizer: AutoProcessor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            Flux2FoveatedUnit_ShapeChecker(),
            Flux2FoveatedUnit_PromptEmbedder(),
            Flux2FoveatedUnit_Qwen3PromptEmbedder(),
            Flux2FoveatedUnit_NoiseInitializer(),
            Flux2FoveatedUnit_InputImageEmbedder(),
            Flux2FoveatedUnit_ImageIDs(),
        ]
        
        # token upsampler and downsampler for foveation
        self.token_upsampler = None
        self.token_downsampler = None

        # fixed latent for tracking validation
        self.fixed_clean_latent = None
        self.detach_z_mix = False

        self.model_fn = model_fn_flux2_foveated
        
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
    ):
        """
        Load a FLUX2 Foveated Image Pipeline from pretrained weights.
        
        The pipeline will first try to fetch a registered "flux2_dit_foveated" model.
        If not found, it will fetch a standard "flux2_dit" model and convert it to
        the foveated version (same weights, different attention processing).
        """
        # Initialize pipeline
        pipe = Flux2FoveatedImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("flux2_text_encoder")
        pipe.text_encoder_qwen3 = model_pool.fetch_model("z_image_text_encoder")
        
        # Try to fetch foveated model first, fall back to base model
        pipe.dit = model_pool.fetch_model("flux2_dit_foveated")
        if pipe.dit is None:
            # Load base FLUX2 model and convert to foveated version
            base_dit = model_pool.fetch_model("flux2_dit")
            if base_dit is not None:
                pipe.dit = Flux2FoveatedImagePipeline._convert_to_foveated_dit(base_dit)
                print("Converted base FLUX2 DiT to foveated version.")
        
        pipe.vae = model_pool.fetch_model("flux2_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @staticmethod
    def _convert_to_foveated_dit(base_dit):
        """
        Convert a base FLUX2 DiT to a foveated version.
        
        The foveated model has the exact same architecture as the base model,
        just with CRPA attention processors. We infer config from state dict.
        
        Args:
            base_dit: A loaded Flux2DiT model instance.
            
        Returns:
            A Flux2DiTFoveated model with the same weights.
        """
        # Get the state dict from the base model
        state_dict = base_dit.state_dict()
        
        # Infer configuration from state dict shapes
        # inner_dim from x_embedder.weight: [inner_dim, in_channels]
        inner_dim = state_dict['x_embedder.weight'].shape[0]
        in_channels = state_dict['x_embedder.weight'].shape[1]
        
        # joint_attention_dim from context_embedder.weight: [inner_dim, joint_attention_dim]
        joint_attention_dim = state_dict['context_embedder.weight'].shape[1]
        
        # Count transformer blocks
        num_layers = 0
        num_single_layers = 0
        for key in state_dict.keys():
            if key.startswith('transformer_blocks.') and key.endswith('.attn.to_q.weight'):
                idx = int(key.split('.')[1])
                num_layers = max(num_layers, idx + 1)
            if key.startswith('single_transformer_blocks.') and key.endswith('.attn.to_qkv_mlp_proj.weight'):
                idx = int(key.split('.')[1])
                num_single_layers = max(num_single_layers, idx + 1)
        
        # Infer attention config from to_q weight: [inner_dim, inner_dim]
        # inner_dim = num_heads * head_dim
        # Default head_dim is 128, so num_heads = inner_dim / 128
        attention_head_dim = 128
        num_attention_heads = inner_dim // attention_head_dim
        
        # Check for guidance embeddings
        guidance_embeds = 'time_guidance_embed.guidance_embedder.linear_1.weight' in state_dict
        
        print(f"Inferred config: inner_dim={inner_dim}, in_channels={in_channels}, "
              f"joint_attention_dim={joint_attention_dim}, num_layers={num_layers}, "
              f"num_single_layers={num_single_layers}, num_attention_heads={num_attention_heads}, "
              f"attention_head_dim={attention_head_dim}, guidance_embeds={guidance_embeds}")
        
        # Create foveated model with inferred parameters
        foveated_dit = Flux2DiTFoveated(
            in_channels=in_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            guidance_embeds=guidance_embeds,
        )
        
        # Load state dict directly - architectures are identical
        foveated_dit.load_state_dict(state_dict)
        
        # Move to same device and dtype as base model
        device = next(base_dit.parameters()).device
        dtype = next(base_dit.parameters()).dtype
        foveated_dit = foveated_dit.to(device=device, dtype=dtype)
        
        return foveated_dit

    # =====================================================================
    # Patchify / Unpatchify Operations (FLUX2 2x2 patches)
    # =====================================================================

    @staticmethod
    def _patchify_latents(latents):
        """Convert [B, C, H, W] to 2x2 patched format [B, C*4, H/2, W/2]."""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents):
        """Convert [B, C*4, H/2, W/2] back to [B, C, H, W]."""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
        return latents

    # =====================================================================
    # Foveation-Specific Latent Operations
    # =====================================================================

    @staticmethod
    def _prepare_latent_image_ids_foveation(
        batch_size: int, 
        height: int, 
        width: int, 
        device: torch.device, 
        dtype: torch.dtype, 
        foveation_mask: Optional[torch.Tensor] = None
    ):
        """
        Prepare 4D position IDs for image latents with foveation support.
        
        For FLUX2, position IDs have format (T, H, W, L) where:
        - T: Time coordinate (always 0 for image latents)
        - H: Height coordinate
        - W: Width coordinate  
        - L: Layer coordinate (always 0 for image latents)
        
        With foveation:
        - High-res blocks keep all 4 tokens
        - Low-res blocks keep only top-left token
        
        Returns:
            latent_image_ids: [Seq_Len, 4] position IDs
            resolution_mask: [Seq_Len] mask (1=HR, 0=LR)
            resolution_mask_top_left: [Seq_Len] mask for key subsampling
        """
        # Create base coordinate grid (all must have same dtype for cartesian_prod)
        t = torch.zeros(1, dtype=dtype)
        h = torch.arange(height, dtype=dtype)
        w = torch.arange(width, dtype=dtype)
        l = torch.zeros(1, dtype=dtype)
        
        latent_image_ids = torch.cartesian_prod(t, h, w, l).to(device=device)
        
        resolution_mask_grid = torch.ones(height, width, device=device, dtype=dtype)

        if foveation_mask is not None:
            if height % 2 != 0 or width % 2 != 0:
                raise ValueError("Height and width must be even for mixed-resolution.")
            
            h_2, w_2 = height // 2, width // 2
            
            grid_2d = latent_image_ids.view(height, width, 4)
            grid_blocks = grid_2d.view(h_2, 2, w_2, 2, 4).permute(0, 2, 1, 3, 4).reshape(h_2, w_2, 4, 4)
            
            mask_blocks = foveation_mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
            is_high_res_block = (mask_blocks.sum(dim=-1) > 0)
            is_low_res_block = ~is_high_res_block

            low_res_coords = grid_blocks[..., 0, :]
            output_grid = grid_blocks.clone()
            output_grid[is_low_res_block, 0, :] = low_res_coords[is_low_res_block]

            res_mask_blocks = resolution_mask_grid.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
            output_res_mask = res_mask_blocks.clone()
            output_res_mask[is_low_res_block, :] = 0.0

            valid_tokens = torch.ones(h_2, w_2, 4, device=device, dtype=torch.bool)
            valid_tokens[is_low_res_block, 1:] = False

            output_grid = output_grid.reshape(-1, 4)
            output_res_mask = output_res_mask.reshape(-1)
            valid_tokens = valid_tokens.reshape(-1)

            tl_mask_blocks = torch.zeros_like(res_mask_blocks)
            tl_mask_blocks[..., 0] = 1.0
            tl_mask_blocks[is_low_res_block, :] = 1.0
            resolution_mask_top_left = tl_mask_blocks.reshape(-1)[valid_tokens]

            latent_image_ids = output_grid[valid_tokens]
            resolution_mask = output_res_mask[valid_tokens]

            return (
                latent_image_ids.to(device=device, dtype=dtype),
                resolution_mask.to(device=device, dtype=dtype),
                resolution_mask_top_left.to(device=device, dtype=dtype)
            )
        else:
            return (
                latent_image_ids.to(device=device, dtype=dtype),
                None,
                None
            )

    @staticmethod
    def _downsample_latents(
        latents: torch.Tensor,
        height: int,
        width: int,
        foveation_mask: Optional[torch.Tensor] = None,
        sample_mode: str = "nearest",
        token_downsampler: Optional[torch.nn.Module] = None,
        temb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        latents_downsampled: Optional[torch.Tensor] = None,
    ):
        """
        Downsample latents spatially based on foveation mask.
        
        For low-res blocks, averages/samples 2x2 regions to single tokens.
        High-res blocks keep all 4 tokens.
        """
        if foveation_mask is None:
            return latents
            
        mask = foveation_mask.to(latents.device)
        batch_size, seq_len, channels = latents.shape

        if height * width != seq_len:
            raise ValueError(f"Height ({height}) * Width ({width}) != seq_len ({seq_len})")

        output_latents = latents.clone()

        # Unpack to spatial format for downsampling
        latents_spatial = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents_spatial = latents_spatial.permute(0, 3, 1, 4, 2, 5)
        latents_spatial = latents_spatial.reshape(batch_size, channels // 4, height * 2, width * 2)

        # Downsample
        if latents_downsampled is None:
            if token_downsampler is not None:
                latents_down = token_downsampler(
                    latents_spatial,
                    temb=temb,
                    timestep=timestep,
                    guidance=guidance,
                    foveation_mask=foveation_mask,
                )
            elif sample_mode == "average":
                latents_down = F.interpolate(latents_spatial, scale_factor=0.5, mode="bilinear") * 2
            elif sample_mode == "top_left":
                latents_down = F.interpolate(latents_spatial, scale_factor=0.5, mode="nearest")
            else:
                raise ValueError(f"Unknown sample_mode: {sample_mode}")

            # Repack
            latents_down = latents_down.reshape(batch_size, channels // 4, height // 2, 2, width // 2, 2)
            latents_down = latents_down.permute(0, 2, 4, 1, 3, 5)
            low_res_packed = latents_down.reshape(batch_size, height // 2, width // 2, channels)
        else:
            low_res_packed = latents_downsampled.reshape(batch_size, height // 2, width // 2, channels)

        h_2, w_2 = height // 2, width // 2
        output_latents = output_latents.view(batch_size, height, width, channels)
        output_latents = output_latents.view(batch_size, h_2, 2, w_2, 2, channels)
        output_latents = output_latents.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, h_2, w_2, 4, channels)

        mask_blocks = mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
        is_high_res_block = (mask_blocks.sum(dim=-1) > 0)
        is_low_res_block = ~is_high_res_block

        output_latents[:, is_low_res_block, 0, :] = low_res_packed[:, is_low_res_block, :].to(output_latents.dtype)

        valid_tokens = torch.ones(h_2, w_2, 4, device=latents.device, dtype=torch.bool)
        valid_tokens[is_low_res_block, 1:] = False

        output_latents = output_latents.view(batch_size, -1, channels)
        valid_tokens = valid_tokens.view(-1)
        
        return output_latents[:, valid_tokens, :]

    @staticmethod
    def _downsample_remaining_high_res(
        latents: torch.Tensor,
        height: int,
        width: int,
        foveation_mask: torch.Tensor,
        downsample_mode: str = "nearest",
    ):
        """
        Downsample all blocks to uniform low-res (single token per 2x2 block).
        """
        batch_size, reduced_len, channels = latents.shape
        device = latents.device
        dtype = latents.dtype

        h_2, w_2 = height // 2, width // 2

        mask = foveation_mask.to(device)
        mask_blocks = mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)

        is_high_res_block = (mask_blocks.sum(dim=-1) > 0)
        is_low_res_block = ~is_high_res_block

        layout_mask = torch.ones(h_2, w_2, 4, device=device, dtype=torch.bool)
        layout_mask[is_low_res_block, 1:] = False

        if layout_mask.sum() != reduced_len:
            raise ValueError(f"Latent length {reduced_len} != mask topology {layout_mask.sum()}")

        reconstructed_blocks = torch.zeros(batch_size, h_2 * w_2 * 4, channels, device=device, dtype=dtype)
        reconstructed_blocks[:, layout_mask.view(-1), :] = latents
        reconstructed_blocks = reconstructed_blocks.view(batch_size, h_2, w_2, 4, channels)

        if is_high_res_block.any():
            high_res_tokens = reconstructed_blocks[:, is_high_res_block, :, :]
            
            if downsample_mode == "average":
                pooled_val = high_res_tokens.mean(dim=2)
                reconstructed_blocks[:, is_high_res_block, 0, :] = pooled_val

        final_grid = reconstructed_blocks[:, :, :, 0, :]
        return final_grid.reshape(batch_size, h_2 * w_2, channels)

    @staticmethod
    def _upsample_latents(
        latents: torch.Tensor,
        height: int,
        width: int,
        foveation_mask: Optional[torch.Tensor] = None,
        upsample_mode: str = "nearest",
    ):
        """
        Upsamples reduced latents back to the original spatial grid size.
        """
        if foveation_mask is None:
            return latents

        batch_size, reduced_len, channels = latents.shape
        device = latents.device
        dtype = latents.dtype
        
        h_2, w_2 = height // 2, width // 2
        
        mask = foveation_mask.to(device)
        mask_blocks = mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
        
        is_high_res_block = (mask_blocks.sum(dim=-1) > 0)
        is_low_res_block = ~is_high_res_block
        
        layout_mask = torch.ones(h_2, w_2, 4, device=device, dtype=torch.bool)
        layout_mask[is_low_res_block, 1:] = False
        
        if layout_mask.sum() != reduced_len:
            raise ValueError(f"Latent length {reduced_len} does not match mask topology {layout_mask.sum()}.")

        reconstructed_blocks = torch.zeros(batch_size, h_2 * w_2 * 4, channels, device=device, dtype=dtype)
        reconstructed_blocks[:, layout_mask.view(-1), :] = latents
        reconstructed_blocks = reconstructed_blocks.view(batch_size, h_2, w_2, 4, channels)

        if upsample_mode == "nearest":
            if is_low_res_block.any():
                low_res_vals = reconstructed_blocks[:, is_low_res_block, 0:1, :]
                reconstructed_blocks[:, is_low_res_block, :, :] = low_res_vals.expand(-1, -1, 4, -1)
                
            final_output = reconstructed_blocks.view(batch_size, h_2, w_2, 2, 2, channels)
            final_output = final_output.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, height, width, channels)
        else:
            coarse_grid = reconstructed_blocks[..., 0, :].clone()
            
            if is_high_res_block.any():
                high_res_means = reconstructed_blocks[:, is_high_res_block, :, :].mean(dim=2)
                coarse_grid[:, is_high_res_block, :] = high_res_means
            
            coarse_grid = coarse_grid.permute(0, 3, 1, 2)
            
            fine_grid = F.interpolate(
                coarse_grid, 
                size=(height, width), 
                mode=upsample_mode, 
                align_corners=False, 
                antialias=True if upsample_mode != 'nearest' else False
            )
            
            final_output = fine_grid.permute(0, 2, 3, 1)
            
            high_res_raster = reconstructed_blocks.view(batch_size, h_2, w_2, 2, 2, channels)
            high_res_raster = high_res_raster.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, height, width, channels)
            
            pixel_mask = is_high_res_block.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
            final_output[:, pixel_mask, :] = high_res_raster[:, pixel_mask, :]

        return final_output.reshape(batch_size, height * width, channels)

    def _save_vlow_x0_vis(
        self,
        v_low_unpacked: torch.Tensor,
        x_0_low: Optional[torch.Tensor],
        save_folder: str,
        progress_id: int,
        prediction_type: str,
    ):
        """Save v_low_unpacked and x_0_low using viridis (value) and inferno (spatial gradient/edges) colormaps."""
        import matplotlib.cm as mpl_cm
        viridis = mpl_cm.get_cmap("viridis")
        inferno = mpl_cm.get_cmap("inferno")

        def _to_scalar(t: torch.Tensor):
            """Reduce first 3 channels to scalar: L2 norm per pixel."""
            x = t[:, :3].float().detach().cpu()
            return (x ** 2).sum(dim=1, keepdim=True).sqrt().squeeze(1)

        def _spatial_gradient_mag(t: torch.Tensor):
            """Compute spatial gradient magnitude (edges). Input [B, C, H, W], output [B, H, W]."""
            x = t[:, :3].float().detach().cpu().mean(dim=1)
            gx = F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 1), mode="replicate")
            gy = F.pad(x[:, 1:] - x[:, :-1], (0, 0, 0, 1), mode="replicate")
            return (gx ** 2 + gy ** 2).sqrt()

        tensors_to_save = [("v_low", v_low_unpacked)]
        if x_0_low is not None and prediction_type == "clean":
            tensors_to_save.append(("x0_low", x_0_low))

        all_scalars = [_to_scalar(t) for _, t in tensors_to_save]
        vmin = float(min(s.min().item() for s in all_scalars))
        vmax = float(max(s.max().item() for s in all_scalars))
        if vmax - vmin < 1e-8:
            vmax = vmin + 1e-8

        os.makedirs(save_folder, exist_ok=True)
        for (name, t), scalar in zip(tensors_to_save, all_scalars):
            for b in range(t.shape[0]):
                s = scalar[b].numpy()
                s_norm = (s - vmin) / (vmax - vmin)
                s_norm = np.clip(s_norm, 0, 1)
                rgb_viridis = (viridis(s_norm)[:, :, :3] * 255).astype(np.uint8)
                img_v = Image.fromarray(rgb_viridis).resize((512, 512), Image.NEAREST)
                path_v = os.path.join(save_folder, f"step_{progress_id:04d}_{name}_b{b}.png")
                img_v.save(path_v)

                grad = _spatial_gradient_mag(t[b:b + 1]).squeeze(0).numpy()
                gmin, gmax = float(grad.min()), float(grad.max())
                if gmax - gmin < 1e-8:
                    gmax = gmin + 1e-8
                grad_norm = np.clip((grad - gmin) / (gmax - gmin), 0, 1)
                rgb_inferno = (inferno(grad_norm)[:, :, :3] * 255).astype(np.uint8)
                img_g = Image.fromarray(rgb_inferno).resize((512, 512), Image.NEAREST)
                path_g = os.path.join(save_folder, f"step_{progress_id:04d}_{name}_grad_b{b}.png")
                img_g.save(path_g)

    def _upsample_x0_pred_unpacked(
        self,
        noise_pred: torch.Tensor,
        latent_height: int,
        latent_width: int,
        foveation_mask: Optional[torch.Tensor] = None,
        token_upsampler: Optional[torch.nn.Module] = None,
        temb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        z_t_high: Optional[torch.Tensor] = None,
        z_t_low: Optional[torch.Tensor] = None,
        prediction_type: str = "clean",
        vis_save_folder: Optional[str] = None,
        progress_id: Optional[int] = None,
        vis_save_steps: Optional[List[int]] = None,
    ):
        """
        Upsample using X0 (clean image) prediction domain.
        
        Algorithm:
        1. For low-res regions:
           - Get v_low from DiT output
           - Compute x0_low = z_t_low - sigma * v_low
           - Upsample x0_low to x0_high
           - Convert back: v_high = (z_t_high - x0_high) / sigma
        2. For high-res regions:
           - Keep original DiT predictions
        """
        if foveation_mask is None:
            batch_size = noise_pred.shape[0]
            channels = noise_pred.shape[-1]
            noise_pred = noise_pred.view(batch_size, latent_height, latent_width, channels).permute(0, 3, 1, 2)
            return self._unpatchify_latents(noise_pred)

        batch_size = noise_pred.shape[0]
        device = noise_pred.device
        dtype = noise_pred.dtype

        pixel_height = latent_height * 2
        pixel_width = latent_width * 2
        channels = noise_pred.shape[-1]

        # LOW-RES REGIONS: X0 Prediction Upsampling
        v_low_packed = self._downsample_remaining_high_res(
            noise_pred, latent_height, latent_width, foveation_mask=foveation_mask
        )

        h_2, w_2 = latent_height // 2, latent_width // 2
        v_low_spatial = v_low_packed.view(batch_size, h_2, w_2, channels).permute(0, 3, 1, 2)
        v_low_unpacked = self._unpatchify_latents(v_low_spatial) # VISUALIZE

        ## sigma calculation
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        elif sigma.dim() == 2:
            sigma = sigma.view(sigma.shape[0], 1, 1, 1)
        sigma_clamped = sigma.clamp(min=1e-6)
        ## sigma calculation end

        if prediction_type == "clean":
            if self.detach_z_mix:
                z_t_low = z_t_low.detach()
            x_0_low = z_t_low - sigma * v_low_unpacked
            x_0_high_upsampled = token_upsampler(x_0_low, temb=temb, timestep=timestep, guidance=guidance, foveation_mask=foveation_mask)
            v_high_from_upsampler = (z_t_high - x_0_high_upsampled) / sigma_clamped
        elif prediction_type == "flow":
            x_0_low = None
            v_high_from_upsampler = token_upsampler(v_low_unpacked, temb=temb, timestep=timestep, guidance=guidance, foveation_mask=foveation_mask)
        elif prediction_type == "refiner":
            # the token "upsampler" is actually a refiner that refines high-res latents
            x_0_low = z_t_low - sigma * v_low_unpacked
            x_0_high_bicubic = F.interpolate(x_0_low, size=(pixel_height, pixel_width), mode='bicubic', align_corners=False)
            v_high_coarse = (z_t_high - x_0_high_bicubic) / sigma_clamped
            v_high_from_upsampler = token_upsampler(v_high_coarse, temb=temb, timestep=timestep, guidance=guidance, foveation_mask=foveation_mask)  # not actually upsampling, just DiT block
        else:
            raise ValueError(f"Invalid prediction type: {prediction_type}")

        if vis_save_folder is not None and progress_id is not None:
            save_steps = set(vis_save_steps) if vis_save_steps is not None else {0}
            if progress_id in save_steps:
                self._save_vlow_x0_vis(
                    v_low_unpacked,
                    x_0_low,
                    vis_save_folder,
                    progress_id,
                    prediction_type,
                )

        # HIGH-RES REGIONS: Keep Original DiT Predictions
        h_2, w_2 = latent_height // 2, latent_width // 2
        mask = foveation_mask
        mask_blocks = mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
        is_high_res_block = (mask_blocks.sum(dim=-1) > 0)
        is_low_res_block = ~is_high_res_block

        layout_mask = torch.ones(h_2, w_2, 4, device=device, dtype=torch.bool)
        layout_mask[is_low_res_block, 1:] = False

        reconstructed_blocks = torch.zeros(
            batch_size, h_2 * w_2 * 4, channels, device=device, dtype=dtype
        )
        reconstructed_blocks[:, layout_mask.view(-1), :] = noise_pred
        reconstructed_blocks = reconstructed_blocks.view(batch_size, h_2, w_2, 4, -1)

        if is_low_res_block.any():
            lr_tokens = reconstructed_blocks[:, is_low_res_block, 0:1, :]
            reconstructed_blocks[:, is_low_res_block, 1:, :] = lr_tokens

        dense_packed_grid = reconstructed_blocks.view(
            batch_size, h_2, w_2, 2, 2, -1
        ).permute(0, 1, 3, 2, 4, 5).reshape(batch_size, latent_height, latent_width, -1)

        v_high_original_spatial = dense_packed_grid.permute(0, 3, 1, 2)
        v_high_original = self._unpatchify_latents(v_high_original_spatial)

        # COMBINE: Use mask to select HR vs LR regions
        foveation_mask_upsampled = F.interpolate(
            foveation_mask.float().unsqueeze(0).unsqueeze(0),
            size=(pixel_height, pixel_width),
            mode='nearest'
        ).squeeze().bool()
        final_output = torch.where(
            foveation_mask_upsampled.to(device),
            v_high_original.to(device),
            v_high_from_upsampler.to(device)
        )

        return final_output

    def foveated_training_forward(self, inputs: dict, timestep: torch.Tensor, timestep_id: int, prediction_type: str, tokenAE_training: bool):
        """
        Single-step foveated forward for training. Used by FoveatedFlowMatchSFTLoss.
        Downsamples latents with foveation, runs DiT, upsamples output to full resolution.
        Returns full-res noise prediction [B, H*W, C] matching training target shape.
        """
        
        foveation_mask = inputs.get("foveation_mask")
        token_upsampler = getattr(self, "token_upsampler", None)
        token_downsampler = getattr(self, "token_downsampler", None)
        if foveation_mask is None or (tokenAE_training and (token_upsampler is None or token_downsampler is None)):
            models = {name: getattr(self, name) for name in self.in_iteration_models}
            return self.model_fn(**models, **inputs, timestep=timestep)
        
        # print some weights of the token_upsampler
        #if token_upsampler is not None:
        #    print("<Flux2FoveatedImagePipeline> ===> Token upsampler weights:")
        #    for name, param in token_upsampler.named_parameters():
        #        if name == "output_proj.weight":
        #            print(param)
                
        
        # fixed clean latent for tracking validation
        if self.fixed_clean_latent is None:
            print("<Flux2FoveatedImagePipeline> ===> Creating fixed clean latent for tracking validation")
            self.fixed_clean_latent = inputs["input_latents"] # clean input latent

        latents = inputs["latents"]
        height = inputs["height"]
        width = inputs["width"]
        latent_height, latent_width = height // 16, width // 16
        batch_size = latents.shape[0]
        foveation_mask = foveation_mask.to(latents.device)

        if tokenAE_training:
            # Get sigma for timestep
            t = timestep.cpu().squeeze() if hasattr(timestep, "cpu") else timestep.squeeze()
            sigma = self.scheduler.sigmas[timestep_id].to(dtype=latents.dtype, device=latents.device)
            sigma = sigma.view(1, 1, 1, 1)

            # timestep and guidance embedding
            temb = get_time_guidance_embedding(self.dit, timestep, inputs["embedded_guidance"], self.torch_dtype, self.device)

            # z_t high-res unpacked
            latents_spatial = latents.view(batch_size, latent_height, latent_width, -1).permute(0, 3, 1, 2)
            z_t_high_unpacked = self._unpatchify_latents(latents_spatial)

            # Downsample latents for DiT
            latents_input = self._downsample_latents(
                latents, latent_height, latent_width,
                foveation_mask=foveation_mask, token_downsampler=token_downsampler, temb=temb,
                timestep=timestep, guidance=inputs.get("embedded_guidance"),
                latents_downsampled=None
            )
            z_t_low_packed = self._downsample_remaining_high_res(
                latents_input, latent_height, latent_width, foveation_mask=foveation_mask
            )
            h_2, w_2 = latent_height // 2, latent_width // 2
            z_t_low_spatial = z_t_low_packed.view(batch_size, h_2, w_2, -1).permute(0, 3, 1, 2)
            z_t_low_unpacked = self._unpatchify_latents(z_t_low_spatial)
        else:
            latents_input = self._downsample_latents(
                latents, latent_height, latent_width,
                foveation_mask=foveation_mask, token_downsampler=None, temb=None,
                timestep=None, guidance=None,
                latents_downsampled=inputs["latents_downsampled"]
            )

        # Prepare image_ids
        latent_ids, resolution_mask, resolution_mask_top_left = self._prepare_latent_image_ids_foveation(
            batch_size, latent_height, latent_width, self.device, latents.dtype, foveation_mask
        )
        if latent_ids.ndim == 2:
            latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        # Update inputs for model_fn
        inputs["latents"] = latents_input
        inputs["image_ids"] = latent_ids
        inputs["resolution_mask"] = resolution_mask
        inputs["resolution_mask_top_left"] = resolution_mask_top_left

        # Forward through DiT (no CFG for training)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        noise_pred = self.model_fn(**models, **inputs, timestep=timestep)

        # Upsample to full resolution
        if tokenAE_training:
            noise_pred = self._upsample_x0_pred_unpacked(
                noise_pred, latent_height, latent_width,
                foveation_mask=foveation_mask, token_upsampler=token_upsampler, temb=temb,
                timestep=timestep, guidance=inputs.get("embedded_guidance"),
                sigma=sigma, z_t_high=z_t_high_unpacked, z_t_low=z_t_low_unpacked,
                prediction_type=prediction_type,
            )
            noise_pred = self._patchify_latents(noise_pred)
            noise_pred = rearrange(noise_pred, "B C H W -> B (H W) C")
        else:
            # do nothing to the noise prediction from the dit
            #print('no tokenAE training', noise_pred.shape)
            pass
        return noise_pred

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        embedded_guidance: float = 4.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        # Progress bar
        progress_bar_cmd = tqdm,
        # Foveation
        foveation_mask: Optional[torch.Tensor] = None,
        token_upsampler: Optional[torch.nn.Module] = None,
        token_downsampler: Optional[torch.nn.Module] = None,
        decode_mode: str = "direct",
        prediction_type: str = "clean",
        tokenAE_training: bool = False,
        soft_foveation_blend: bool = False,
        vis_vlow_x0_folder: Optional[str] = None,
        vis_vlow_x0_steps: Optional[List[int]] = None,
        full_res_foveation_mask: Optional[torch.Tensor] = None,
    ):
        """
        Generate images with optional foveated rendering.
        
        Args:
            prompt: Text prompt for generation.
            foveation_mask: [H, W] binary mask (1=high-res, 0=low-res regions).
                Shape should match latent dimensions (height//16, width//16).
            token_upsampler: Learned upsampler for x0 prediction upsampling.
            token_downsampler: Learned downsampler for latent downsampling.
            decode_mode: "direct" or "merge" for VAE decoding.
            soft_foveation_blend: If True, apply Gaussian falloff to the foveation mask at merge
                so the blend is soft-edged over ~upscale_factor pixels (smooth transition).
        """
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=height//16*width//16)

        if vis_vlow_x0_folder is not None and vis_vlow_x0_steps is None:
            n = len(self.scheduler.timesteps)
            vis_vlow_x0_steps = [0, n // 4, n // 2, 3 * n // 4, n - 1] if n >= 5 else list(range(n))

        # Prepare latent dimensions
        latent_height = height // 16
        latent_width = width // 16

        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale, "embedded_guidance": embedded_guidance,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "foveation_mask": foveation_mask,
            "soft_foveation_blend": soft_foveation_blend,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise with foveation support
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        
        latents = inputs_shared["latents"]
        if not tokenAE_training:
            # if no tokenAE training, all downsampling is done in the first step only
            latents = self._downsample_latents(
                latents, latent_height, latent_width,
                foveation_mask=foveation_mask, token_downsampler=None, temb=None,
                timestep=None, guidance=None,
                latents_downsampled=inputs_shared["latents_downsampled"]
            )
        print('decode mode', decode_mode)
        
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep_val = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            batch_size = latents.shape[0]

            # Prepare inputs for transformer
            if foveation_mask is not None:
                if tokenAE_training:
                    # time guidance embedding
                    temb = get_time_guidance_embedding(self.dit, timestep_val, embedded_guidance, self.torch_dtype, self.device)

                    latents_high_res_step = latents.clone()
                    
                    # Get sigma for x0 prediction
                    sigma = self.scheduler.sigmas[progress_id].to(dtype=latents.dtype, device=self.device)
                    sigma = sigma.view(1, 1, 1, 1)

                    latents_spatial = latents_high_res_step.view(
                        batch_size, latent_height, latent_width, -1
                    ).permute(0, 3, 1, 2)
                    z_t_high_unpacked = self._unpatchify_latents(latents_spatial)
                    

                    # Downsample latents for DiT input
                    latents_input = self._downsample_latents(
                        latents_high_res_step,
                        latent_height,
                        latent_width,
                        foveation_mask=foveation_mask,
                        token_downsampler=token_downsampler,
                        temb=temb,
                        timestep=timestep_val,
                        guidance=embedded_guidance,
                    )

                    # Get low-res z_t for x0 prediction
                    z_t_low_packed = self._downsample_remaining_high_res(
                        latents_input,
                        latent_height,
                        latent_width,
                        foveation_mask=foveation_mask,
                    )
                    h_2, w_2 = latent_height // 2, latent_width // 2
                    z_t_low_spatial = z_t_low_packed.view(batch_size, h_2, w_2, -1).permute(0, 3, 1, 2)
                    z_t_low_unpacked = self._unpatchify_latents(z_t_low_spatial)
                else:
                    latents_input = latents

                # Prepare position IDs for foveated latents
                latent_ids, resolution_mask, resolution_mask_top_left = self._prepare_latent_image_ids_foveation(
                    batch_size,
                    latent_height,
                    latent_width,
                    self.device,
                    latents.dtype,
                    foveation_mask,
                )
                if latent_ids.ndim == 2:
                    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                latents_input = latents
                z_t_low_unpacked = None
                resolution_mask = None
                resolution_mask_top_left = None
                latent_ids = inputs_shared["image_ids"]

            # Prepare shared inputs for CFG-guided model call
            inputs_shared["latents"] = latents_input
            inputs_shared["image_ids"] = latent_ids
            inputs_shared["resolution_mask"] = resolution_mask
            inputs_shared["resolution_mask_top_left"] = resolution_mask_top_left

            # Forward through transformer with classifier-free guidance
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn,
                cfg_scale,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                **models,
                timestep=timestep_val,
                progress_id=progress_id,
            )

            # Upsample using x0 prediction if foveating
            if foveation_mask is not None:
                if tokenAE_training:
                    noise_pred = self._upsample_x0_pred_unpacked(
                        noise_pred,
                        latent_height,
                        latent_width,
                        foveation_mask=foveation_mask,
                        token_upsampler=token_upsampler,
                        temb=temb,
                        timestep=timestep_val,
                        guidance=embedded_guidance,
                        sigma=sigma,
                        z_t_high=z_t_high_unpacked,
                        z_t_low=z_t_low_unpacked,
                        prediction_type=prediction_type,
                        vis_save_folder=vis_vlow_x0_folder,
                        progress_id=progress_id,
                        vis_save_steps=vis_vlow_x0_steps,
                    )
                    # Pack back
                    noise_pred = self._patchify_latents(noise_pred)
                    noise_pred = rearrange(noise_pred, "B C H W -> B (H W) C")
                else:
                    pass
            else:
                # Slice to latent size only
                noise_pred = noise_pred[:, :latents.size(1), :]

            # Scheduler step
            if tokenAE_training:
                latents = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, latents=latents_high_res_step)
            else:
                latents = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, latents=latents)
        
        inputs_shared["latents"] = latents
        
        # Decode
        self.load_models_to_device(['vae'])
        
        if decode_mode == "direct":
            latents_spatial = rearrange(latents, "B (H W) C -> B C H W", H=latent_height, W=latent_width)
            image = self.vae.decode(latents_spatial)
            image = self.vae_output_to_image(image)
        elif decode_mode == "merge" and foveation_mask is not None:
            # Merge decoding: decode low-res and high-res separately, then merge
            # Note: DiffSynth VAE expects patched latents [B, 128, H/2, W/2] - do NOT unpatchify
            
            # Step 1: Convert full-res latents to mixed-res latents
            if tokenAE_training:
                mixed_res_latents = self._downsample_latents(
                    latents, latent_height, latent_width, 
                    foveation_mask=foveation_mask, token_downsampler=token_downsampler, temb=None
                )
            else:
                mixed_res_latents = latents # already
            
            # Step 2: Low-res path - downsample mixed-res to uniform low-res grid
            low_res_latents = self._downsample_remaining_high_res(
                mixed_res_latents, latent_height, latent_width, foveation_mask=foveation_mask
            )
            h_2, w_2 = latent_height // 2, latent_width // 2
            # VAE expects patched format [B, 128, H/2, W/2] - just reshape, don't unpatchify
            low_res_spatial = low_res_latents.view(batch_size, h_2, w_2, -1).permute(0, 3, 1, 2)
            low_res_image = self.vae.decode(low_res_spatial)

            # Step 3: High-res path - upsample mixed-res to uniform high-res grid
            high_res_latents = self._upsample_latents(
                mixed_res_latents, latent_height, latent_width, foveation_mask=foveation_mask
            )
            # VAE expects patched format [B, 128, H/2, W/2] - just reshape, don't unpatchify
            high_res_spatial = high_res_latents.view(
                batch_size, latent_height, latent_width, -1
            ).permute(0, 3, 1, 2)
            high_res_image = self.vae.decode(high_res_spatial)

            # Step 4: Upsample low-res image and merge
            low_res_image = F.interpolate(low_res_image, size=(height, width), mode='bicubic', align_corners=False)
            
            if full_res_foveation_mask is not None:
                high_res_foveation_mask = full_res_foveation_mask[None, None, :, :].float()
            else:
                high_res_foveation_mask = F.interpolate(
                    foveation_mask[None, None, :, :].float(), 
                    size=(height, width), 
                    mode='bicubic'
                )
            high_res_foveation_mask = high_res_foveation_mask.to(latents.dtype).to(latents.device)
                
            if soft_foveation_blend:
                # Soft edge: Gaussian falloff over ~upscale_factor pixels at high-res
                upscale = height / latent_height
                high_res_foveation_mask = _gaussian_blur_mask_2d(
                    high_res_foveation_mask, sigma_pixels=upscale, device=latents.device, dtype=latents.dtype
                )
                high_res_foveation_mask = high_res_foveation_mask.clamp(0.0, 1.0)
            
            merged_image = low_res_image * (1 - high_res_foveation_mask) + high_res_image * high_res_foveation_mask
            image = self.vae_output_to_image(merged_image)
        else:
            # Fallback to direct
            latents_spatial = rearrange(latents, "B (H W) C -> B C H W", H=latent_height, W=latent_width)
            image = self.vae.decode(latents_spatial)
            image = self.vae_output_to_image(image)
            
        self.load_models_to_device([])

        return image


# =====================================================================
# Pipeline Units (adapted from flux2_image.py)
# =====================================================================

class Flux2FoveatedUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: Flux2FoveatedImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class Flux2FoveatedUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )
        self.system_message = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."

    def format_text_input(self, prompts: List[str], system_message: str = None):
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]
        return [
            [
                {"role": "system", "content": [{"type": "text", "text": system_message}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]

    def get_mistral_3_small_prompt_embeds(
        self,
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        system_message: str = None,
        hidden_states_layers: List[int] = (10, 20, 30),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        messages_batch = self.format_text_input(prompts=prompt, system_message=system_message)
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds
    
    def prepare_text_ids(self, x: torch.Tensor, t_coord: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        out_ids = []
        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)
        return torch.stack(out_ids)

    def encode_prompt(
        self,
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]],
        dtype = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompt_embeds = self.get_mistral_3_small_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                dtype=dtype,
                device=device,
                max_sequence_length=max_sequence_length,
                system_message=self.system_message,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self.prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    def process(self, pipe: Flux2FoveatedImagePipeline, prompt):
        if pipe.text_encoder_qwen3 is not None:
            return {}
        
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds, text_ids = self.encode_prompt(
            pipe.text_encoder, pipe.tokenizer, prompt,
            dtype=pipe.torch_dtype, device=pipe.device,
        )
        return {"prompt_embeds": prompt_embeds, "text_ids": text_ids}


class Flux2FoveatedUnit_Qwen3PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder_qwen3",)
        )
        self.hidden_states_layers = (9, 18, 27)

    def get_qwen3_prompt_embeds(
        self,
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )
            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        with torch.inference_mode():
            output = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        out = torch.stack([output.hidden_states[k] for k in self.hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
        return prompt_embeds

    def prepare_text_ids(self, x: torch.Tensor, t_coord: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        out_ids = []
        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)
        return torch.stack(out_ids)

    def encode_prompt(
        self,
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]],
        dtype = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompt_embeds = self.get_qwen3_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                dtype=dtype,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self.prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    def process(self, pipe: Flux2FoveatedImagePipeline, prompt):
        if pipe.text_encoder_qwen3 is None:
            return {}
        
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds, text_ids = self.encode_prompt(
            pipe.text_encoder_qwen3, pipe.tokenizer, prompt,
            dtype=pipe.torch_dtype, device=pipe.device,
        )
        return {"prompt_embeds": prompt_embeds, "text_ids": text_ids}


class Flux2FoveatedUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise", "downsampled_noise"),
        )

    def process(self, pipe: Flux2FoveatedImagePipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 128, height//16, width//16), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        noise_downsampled = pipe.generate_noise((1, 128, height//16//2, width//16//2), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        noise = noise.reshape(1, 128, height//16 * width//16).permute(0, 2, 1)
        noise_downsampled = noise_downsampled.reshape(1, 128, height//16//2 * width//16//2).permute(0, 2, 1)
        return {"noise": noise, "noise_downsampled": noise_downsampled}


class Flux2FoveatedUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "noise_downsampled"),
            output_params=("latents", "latents_downsampled", "input_latents", "input_latents_downsampled"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: Flux2FoveatedImagePipeline, input_image, noise, noise_downsampled):
        if input_image is None:
            return {
                "latents": noise, 
                "latents_downsampled": noise_downsampled,
                "input_latents": None, 
                "input_latents_downsampled": None
            }
        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(input_image)
        input_latents = pipe.vae.encode(image)
        _, _, latent_height, latent_width = input_latents.shape
        input_latents = rearrange(input_latents, "B C H W -> B (H W) C")

        # create downsample 
        image_downsampled = F.interpolate(image, size=(image.shape[-2]//2, image.shape[-1]//2), mode='bicubic', align_corners=False)
        input_latents_downsampled = pipe.vae.encode(image_downsampled)
        input_latents_downsampled = rearrange(input_latents_downsampled, "B C H W -> B (H W) C") # 4X less tokens


        if pipe.scheduler.training:
            return {
                "latents": noise, 
                "latents_downsampled": noise_downsampled,
                "input_latents": input_latents,
                "input_latents_downsampled": input_latents_downsampled
            }
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            latents_downsampled = pipe.scheduler.add_noise(input_latents_downsampled, noise_downsampled, timestep=pipe.scheduler.timesteps[0])
            return {
                "latents": latents, 
                "latents_downsampled": latents_downsampled, 
                "input_latents": input_latents, 
                "input_latents_downsampled": input_latents_downsampled
            }


class Flux2FoveatedUnit_ImageIDs(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "foveation_mask"),
            output_params=("image_ids",),
        )

    def prepare_latent_ids(self, height, width, device, dtype, foveation_mask=None):
        return Flux2FoveatedImagePipeline._prepare_latent_image_ids_foveation(
            batch_size=1,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
            foveation_mask=foveation_mask
        )

    def process(self, pipe: Flux2FoveatedImagePipeline, height, width, foveation_mask):
        latent_ids, resolution_mask, resolution_mask_top_left = self.prepare_latent_ids(
            height // 16, width // 16, pipe.device, pipe.torch_dtype, foveation_mask
        )
        latent_ids = latent_ids.unsqueeze(0).expand(1, -1, -1)
        return {
            "image_ids": latent_ids,
            "resolution_mask": resolution_mask,
            "resolution_mask_top_left": resolution_mask_top_left,
        }


# =====================================================================
# Model Function for Foveated FLUX2
# =====================================================================

def model_fn_flux2_foveated(
    dit: Flux2DiTFoveated,
    latents=None,
    timestep=None,
    embedded_guidance=None,
    prompt_embeds=None,
    text_ids=None,
    image_ids=None,
    resolution_mask=None,
    resolution_mask_top_left=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    """Model function for foveated FLUX2 generation."""
    image_seq_len = latents.shape[1]
    embedded_guidance = torch.tensor([embedded_guidance], device=latents.device)
    
    model_output = dit(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=embedded_guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=image_ids,
        resolution_mask=resolution_mask,
        resolution_mask_top_left=resolution_mask_top_left,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    
    model_output = model_output[:, :image_seq_len]
    return model_output

def get_time_guidance_embedding(dit, timestep, guidance, dtype, device):
    timestep = timestep.to(dtype=dtype, device=device)
    if guidance is not None:
        guidance = torch.tensor([guidance]).to(dtype=dtype, device=device) * 1000
    # 1000, 4000

    #print("in pipeline: timestep: ", timestep, "guidance: ", guidance)
    return dit.time_guidance_embed(timestep, guidance)