# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FLUX2 Klein Pipeline with Foveated Generation and X0 Prediction Upsampling.

This pipeline integrates foveation logic from pipeline_flux_foveation_x0pred.py 
into the FLUX2 Klein architecture.

Key features:
- Mixed-resolution latent processing for foveated rendering
- X0 (clean image) prediction for upsampling low-res regions
- Phase-aligned RoPE for consistent position encoding
- CRPA attention for efficient mixed-resolution attention
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from diffusers.loaders import Flux2LoraLoaderMixin
from diffusers.models import AutoencoderKLFlux2
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput

from transformer_flux2_foveation import Flux2PhaseAlignedTransformer2DModel


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from pipeline_flux2_klein_foveation import Flux2KleinFoveationPipeline

        >>> pipe = Flux2KleinFoveationPipeline.from_pretrained(
        ...     "black-forest-labs/FLUX.2-klein-base-9B", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Create a foveation mask (1=high-res, 0=low-res)
        >>> foveation_mask = torch.zeros(64, 64)
        >>> foveation_mask[16:48, 16:48] = 1  # Center region high-res
        >>> image = pipe(prompt, foveation_mask=foveation_mask, num_inference_steps=50).images[0]
        >>> image.save("flux2_foveated.png")
        ```
"""


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute mu parameter for timestep scheduling."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Retrieve timesteps from scheduler."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    """Retrieve latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class Flux2KleinFoveationPipeline(DiffusionPipeline, Flux2LoraLoaderMixin):
    """
    FLUX2 Klein pipeline for text-to-image generation with foveated rendering.
    
    This pipeline implements:
    - Mixed-resolution latent processing for efficient foveated rendering
    - X0 (clean image) prediction for better upsampling quality
    - Phase-aligned RoPE for consistent position encoding across resolutions
    
    Args:
        transformer ([`Flux2PhaseAlignedTransformer2DModel`]):
            Phase-aligned Transformer for foveated denoising.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Scheduler for denoising.
        vae ([`AutoencoderKLFlux2`]):
            VAE for encoding/decoding images.
        text_encoder ([`Qwen3ForCausalLM`]):
            Qwen3 text encoder.
        tokenizer (`Qwen2TokenizerFast`):
            Tokenizer for text processing.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        transformer: Flux2PhaseAlignedTransformer2DModel,
        is_distilled: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )

        self.register_to_config(is_distilled=is_distilled)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

    # =====================================================================
    # Text Encoding
    # =====================================================================

    @staticmethod
    def _get_qwen3_prompt_embeds(
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        hidden_states_layers: List[int] = (9, 18, 27),
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

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (9, 18, 27),
    ):
        device = device or self._execution_device

        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self._get_qwen3_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    # =====================================================================
    # Position ID Preparation (FLUX2 4D format: T, H, W, L)
    # =====================================================================

    @staticmethod
    def _prepare_text_ids(x: torch.Tensor, t_coord: Optional[torch.Tensor] = None):
        """Prepare 4D position IDs for text tokens."""
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
        - Low-res blocks keep only top-left token with averaged coordinates
        
        Returns:
            latent_image_ids: [Seq_Len, 4] position IDs
            resolution_mask: [Seq_Len] boolean mask (True=HR, False=LR)
            resolution_mask_top_left: [Seq_Len] boolean mask for key subsampling
        """
        # Create base coordinate grid (all must have same dtype for cartesian_prod)
        t = torch.zeros(1, dtype=dtype)  # Time = 0
        h = torch.arange(height, dtype=dtype)
        w = torch.arange(width, dtype=dtype)
        l = torch.zeros(1, dtype=dtype)  # Layer = 0
        
        # Create full grid: [H*W, 4] and move to device
        latent_image_ids = torch.cartesian_prod(t, h, w, l).to(device=device)  # [H*W, 4]
        
        # Resolution mask (1=HR, 0=LR)
        resolution_mask_grid = torch.ones(height, width, device=device, dtype=dtype)

        if foveation_mask is not None:
            if height % 2 != 0 or width % 2 != 0:
                raise ValueError("Height and width must be even for mixed-resolution stride 2.")
            
            h_2, w_2 = height // 2, width // 2
            
            # Reshape to 2x2 blocks: [H/2, W/2, 4, 4] for coordinates
            grid_2d = latent_image_ids.view(height, width, 4)
            grid_blocks = grid_2d.view(h_2, 2, w_2, 2, 4).permute(0, 2, 1, 3, 4).reshape(h_2, w_2, 4, 4)
            
            # Mask blocks: [H/2, W/2, 4]
            mask_blocks = foveation_mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
            is_high_res_block = (mask_blocks.sum(dim=-1) > 0)  # [H/2, W/2]
            is_low_res_block = ~is_high_res_block

            # For LR blocks, use top-left coordinates
            low_res_coords = grid_blocks[..., 0, :]  # [H/2, W/2, 4]
            output_grid = grid_blocks.clone()
            output_grid[is_low_res_block, 0, :] = low_res_coords[is_low_res_block]

            # Resolution mask
            res_mask_blocks = resolution_mask_grid.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
            output_res_mask = res_mask_blocks.clone()
            output_res_mask[is_low_res_block, :] = 0.0

            # Valid tokens (LR blocks only keep index 0)
            valid_tokens = torch.ones(h_2, w_2, 4, device=device, dtype=torch.bool)
            valid_tokens[is_low_res_block, 1:] = False

            output_grid = output_grid.reshape(-1, 4)
            output_res_mask = output_res_mask.reshape(-1)
            valid_tokens = valid_tokens.reshape(-1)

            # Top-left mask for key subsampling
            tl_mask_blocks = torch.zeros_like(res_mask_blocks)
            tl_mask_blocks[..., 0] = 1.0  # Top-left
            tl_mask_blocks[is_low_res_block, :] = 1.0  # All LR tokens
            resolution_mask_top_left = tl_mask_blocks.reshape(-1)[valid_tokens]

            latent_image_ids = output_grid[valid_tokens]  # [Seq_Len, 4]
            resolution_mask = output_res_mask[valid_tokens]  # [Seq_Len]

            return (
                latent_image_ids.to(device=device, dtype=dtype),
                resolution_mask.to(device=device, dtype=dtype),
                resolution_mask_top_left.to(device=device, dtype=dtype)
            )
        else:
            # Standard path without foveation
            return (
                latent_image_ids.to(device=device, dtype=dtype),
                None,
                None
            )

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor):
        """Standard latent ID preparation for FLUX2."""
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0]
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)  # [0]

        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    # =====================================================================
    # Patchify / Unpatchify (FLUX2 2x2 patches)
    # =====================================================================

    @staticmethod
    def _patchify_latents(latents):
        """Convert [B, C, H, W] to 2x2 patched format [B, C*4, H/2, W/2]."""
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents):
        """Convert [B, C*4, H/2, W/2] back to [B, C, H, W]."""
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @staticmethod
    def _pack_latents(latents):
        """Pack latents: [B, C, H, W] -> [B, H*W, C]."""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @staticmethod
    def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        """Unpack latents using position IDs."""
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    # =====================================================================
    # Foveation-Specific Latent Operations
    # =====================================================================

    @staticmethod
    def _downsample_latents(
        latents: torch.Tensor,
        height: int,
        width: int,
        foveation_mask: Optional[torch.Tensor] = None,
        sample_mode: str = "average",
        token_downsampler: Optional[torch.nn.Module] = None,
        temb: Optional[torch.Tensor] = None,
    ):
        """
        Downsample latents spatially based on foveation mask.
        
        For low-res blocks, averages/samples 2x2 regions to single tokens.
        High-res blocks keep all 4 tokens.
        
        Args:
            latents: [B, Seq_Len, C] packed latents
            height, width: Dimensions of latent grid
            foveation_mask: [H, W] binary mask (1=HR, 0=LR)
            
        Returns:
            [B, Reduced_Seq_Len, C] downsampled latents
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
        if token_downsampler is not None:
            latents_down = token_downsampler(latents_spatial, temb=temb, foveation_mask=foveation_mask)
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
        downsample_mode: str = "average",
    ):
        """
        Downsample all blocks to uniform low-res (single token per 2x2 block).
        
        Takes mixed-resolution latents and creates a uniform coarse grid
        by averaging/sampling high-res blocks.
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
            # "nearest" keeps index 0 as-is

        final_grid = reconstructed_blocks[:, :, :, 0, :]
        return final_grid.reshape(batch_size, h_2 * w_2, channels)

    @staticmethod
    def _upsample_latents(
        latents: torch.Tensor,
        height: int,
        width: int,
        foveation_mask: Optional[torch.Tensor] = None,
        upsample_mode: str = "bilinear",
    ):
        """
        Upsamples reduced latents back to the original spatial grid size.
        
        Args:
            latents: [Batch, Reduced_Seq_Len, Channels]
            height: Original grid height.
            width: Original grid width.
            foveation_mask: [height, width] Tensor used for downsampling.
            upsample_mode: "nearest", "bilinear", or "bicubic". 
                           "nearest" duplicates tokens (original behavior).
                           "bilinear"/"bicubic" interpolates low-res regions.
        """
        if foveation_mask is None:
            return latents

        batch_size, reduced_len, channels = latents.shape
        device = latents.device
        dtype = latents.dtype
        
        h_2, w_2 = height // 2, width // 2
        
        # 1. Determine Block Status
        mask = foveation_mask.to(device)
        # [H, W] -> [H/2, W/2, 4]
        mask_blocks = mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
        
        is_high_res_block = (mask_blocks.sum(dim=-1) > 0)  # [H/2, W/2]
        is_low_res_block = ~is_high_res_block
        
        # 2. Scatter Reduced Latents into Block Structure
        # Create Layout Mask (same as downsample logic)
        layout_mask = torch.ones(h_2, w_2, 4, device=device, dtype=torch.bool)
        layout_mask[is_low_res_block, 1:] = False
        
        # Validation
        if layout_mask.sum() != reduced_len:
            raise ValueError(f"Latent length {reduced_len} does not match mask topology {layout_mask.sum()}.")

        # Scatter into sparse buffer: [B, H/2, W/2, 4, C]
        reconstructed_blocks = torch.zeros(batch_size, h_2 * w_2 * 4, channels, device=device, dtype=dtype)
        reconstructed_blocks[:, layout_mask.view(-1), :] = latents
        reconstructed_blocks = reconstructed_blocks.view(batch_size, h_2, w_2, 4, channels)

        # 3. Handle Upsampling
        if upsample_mode == "nearest":
            # Original logic: Duplicate index 0 for low-res blocks
            if is_low_res_block.any():
                low_res_vals = reconstructed_blocks[:, is_low_res_block, 0:1, :]
                reconstructed_blocks[:, is_low_res_block, :, :] = low_res_vals.expand(-1, -1, 4, -1)
                
            # Flatten blocks to raster
            final_output = reconstructed_blocks.view(batch_size, h_2, w_2, 2, 2, channels)
            final_output = final_output.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, height, width, channels)

        else:
            # --- Bilinear / Bicubic Path ---
            
            # A. Create a Dense Coarse Grid [B, C, H/2, W/2]
            coarse_grid = reconstructed_blocks[..., 0, :].clone()  # [B, H/2, W/2, C]
            
            # If block is high-res, replace with mean of all 4 tokens to prevent artifacts
            if is_high_res_block.any():
                high_res_means = reconstructed_blocks[:, is_high_res_block, :, :].mean(dim=2)
                coarse_grid[:, is_high_res_block, :] = high_res_means
            
            # Rearrange for interpolation: [B, C, H/2, W/2]
            coarse_grid = coarse_grid.permute(0, 3, 1, 2)
            
            # B. Interpolate to Full Res [B, C, H, W]
            fine_grid = F.interpolate(
                coarse_grid, 
                size=(height, width), 
                mode=upsample_mode, 
                align_corners=False, 
                antialias=True if upsample_mode != 'nearest' else False
            )
            
            # Back to [B, H, W, C]
            final_output = fine_grid.permute(0, 2, 3, 1)
            
            # C. Paste Original High-Res Details
            high_res_raster = reconstructed_blocks.view(batch_size, h_2, w_2, 2, 2, channels)
            high_res_raster = high_res_raster.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, height, width, channels)
            
            # Expand boolean mask to full pixel resolution [H, W]
            pixel_mask = is_high_res_block.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
            
            # Overwrite high-res pixels
            final_output[:, pixel_mask, :] = high_res_raster[:, pixel_mask, :]

        # Final flatten to sequence
        return final_output.reshape(batch_size, height * width, channels)

    @staticmethod
    def _upsample_x0_pred_unpacked(
        noise_pred: torch.Tensor,
        latent_height: int,
        latent_width: int,
        foveation_mask: Optional[torch.Tensor] = None,
        token_upsampler: Optional[torch.nn.Module] = None,
        temb: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        z_t_high: Optional[torch.Tensor] = None,
        z_t_low: Optional[torch.Tensor] = None,
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
           
        Returns:
            [B, C, H, W] - Full resolution velocity in unpacked format
        """
        if foveation_mask is None:
            # No foveation - just unpack using provided dimensions
            batch_size = noise_pred.shape[0]
            channels = noise_pred.shape[-1]
            # Use provided latent_height and latent_width instead of assuming square
            noise_pred = noise_pred.view(batch_size, latent_height, latent_width, channels).permute(0, 3, 1, 2)
            return Flux2KleinFoveationPipeline._unpatchify_latents(noise_pred)

        batch_size = noise_pred.shape[0]
        device = noise_pred.device
        dtype = noise_pred.dtype

        pixel_height = latent_height * 2
        pixel_width = latent_width * 2
        channels = noise_pred.shape[-1]

        # =====================================================================
        # LOW-RES REGIONS: X0 Prediction Upsampling
        # =====================================================================

        # 1. Get uniform low-res velocity
        v_low_packed = Flux2KleinFoveationPipeline._downsample_remaining_high_res(
            noise_pred, latent_height, latent_width, foveation_mask=foveation_mask
        )

        # 2. Unpack to spatial: [B, C, H/2, W/2]
        h_2, w_2 = latent_height // 2, latent_width // 2
        v_low_spatial = v_low_packed.view(batch_size, h_2, w_2, channels).permute(0, 3, 1, 2)
        v_low_unpacked = Flux2KleinFoveationPipeline._unpatchify_latents(v_low_spatial)

        # 3. Convert velocity to clean image: x0 = z_t - sigma * v
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        elif sigma.dim() == 2:
            sigma = sigma.view(sigma.shape[0], 1, 1, 1)

        x_0_low = z_t_low - sigma * v_low_unpacked

        # 4. Upsample clean image prediction
        x_0_high_upsampled = token_upsampler(x_0_low, temb=temb, foveation_mask=foveation_mask)

        # 5. Convert back to velocity
        sigma_clamped = sigma.clamp(min=1e-6)
        v_high_from_upsampler = (z_t_high - x_0_high_upsampled) / sigma_clamped

        # =====================================================================
        # HIGH-RES REGIONS: Keep Original DiT Predictions
        # =====================================================================

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

        # Fill LR holes with nearest neighbor
        if is_low_res_block.any():
            lr_tokens = reconstructed_blocks[:, is_low_res_block, 0:1, :]
            reconstructed_blocks[:, is_low_res_block, 1:, :] = lr_tokens

        # View as spatial grid
        dense_packed_grid = reconstructed_blocks.view(
            batch_size, h_2, w_2, 2, 2, -1
        ).permute(0, 1, 3, 2, 4, 5).reshape(batch_size, latent_height, latent_width, -1)

        # Convert to [B, C, H, W] format
        v_high_original_spatial = dense_packed_grid.permute(0, 3, 1, 2)
        v_high_original = Flux2KleinFoveationPipeline._unpatchify_latents(v_high_original_spatial)

        # =====================================================================
        # COMBINE: Use mask to select HR vs LR regions
        # =====================================================================

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

    # =====================================================================
    # VAE Operations
    # =====================================================================

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")

        image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        image_latents = self._patchify_latents(image_latents)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std

        return image_latents

    # =====================================================================
    # Latent Preparation
    # =====================================================================

    def prepare_latents(
        self,
        batch_size: int,
        num_latents_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
        foveation_mask: Optional[torch.Tensor] = None,
    ):
        """Prepare initial noise latents with foveation support."""
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        self.latent_height = height // 2
        self.latent_width = width // 2

        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # Prepare position IDs with foveation
        latent_ids, resolution_mask, resolution_mask_top_left = self._prepare_latent_image_ids_foveation(
            batch_size, self.latent_height, self.latent_width, device, dtype, foveation_mask=foveation_mask
        )

        # Expand for batch
        if latent_ids.ndim == 2:
            latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        # Pack latents
        latents = self._pack_latents(latents)

        return latents, latent_ids, resolution_mask, resolution_mask_top_left

    # =====================================================================
    # Input Validation
    # =====================================================================

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale=None,
    ):
        if (
            height is not None and height % (self.vae_scale_factor * 2) != 0
            or width is not None and width % (self.vae_scale_factor * 2) != 0
        ):
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if guidance_scale > 1.0 and self.config.is_distilled:
            logger.warning(f"Guidance scale {guidance_scale} is ignored for distilled models.")

    # =====================================================================
    # Properties
    # =====================================================================

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and not self.config.is_distilled

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    # =====================================================================
    # Main Generation
    # =====================================================================

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[Union[str, List[str]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (9, 18, 27),
        # Foveation-specific parameters
        foveation_mask: Optional[torch.Tensor] = None,
        token_upsampler: Optional[torch.nn.Module] = None,
        token_downsampler: Optional[torch.nn.Module] = None,
        decode_mode: str = "direct",
    ):
        r"""
        Generate images with optional foveated rendering.
        
        Args:
            prompt: Text prompt(s) for generation.
            height, width: Output image dimensions.
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.
            foveation_mask: [H, W] binary mask (1=high-res, 0=low-res regions).
                Shape should match latent dimensions (height//16, width//16).
            token_upsampler: Learned upsampler for x0 prediction upsampling.
            token_downsampler: Learned downsampler for latent downsampling.
            decode_mode: VAE decoding mode - "direct" (full latents) or "merge" 
                (separate low-res and high-res decoding then merge).
            
        Returns:
            Generated images.
            
        Examples:
        """
        # 1. Check inputs
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode prompt
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        # 4. Handle CFG
        if self.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        # 5. Set dimensions
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 6. Prepare latents
        num_channels_latents = self.transformer.config.in_channels // 4
        latents_high_res, latent_ids, resolution_mask, resolution_mask_top_left = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
            foveation_mask=None,  # Start with full resolution
        )

        # Store resolution mask in attention kwargs
        self._attention_kwargs["resolution_mask"] = resolution_mask
        self._attention_kwargs["resolution_mask_top_left"] = resolution_mask_top_left

        # 7. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents_high_res.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Helper function to get sigmas
        def get_sigmas(timesteps_input, n_dim=4, dtype=torch.float32):
            scheduler_sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
            schedule_timesteps = self.scheduler.timesteps.to(device)
            timesteps_input = timesteps_input.to(device)

            step_indices = []
            for t_val in timesteps_input:
                idx = (schedule_timesteps == t_val).nonzero()
                if idx.numel() > 0:
                    step_indices.append(idx.item())
                else:
                    idx = (schedule_timesteps - t_val).abs().argmin().item()
                    step_indices.append(idx)

            sigma = scheduler_sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        # 8. Denoising loop
        self.scheduler.set_begin_index(0)
        latents = latents_high_res

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latents_high_res_step = latents.clone()
                
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # Get sigma for x0 prediction
                sigma = get_sigmas(t.unsqueeze(0), n_dim=4, dtype=latents.dtype)

                # Get high-res z_t in unpacked format
                latents_spatial = latents_high_res_step.view(
                    batch_size, self.latent_height, self.latent_width, -1
                ).permute(0, 3, 1, 2)
                z_t_high_unpacked = self._unpatchify_latents(latents_spatial)

                # Prepare foveation if mask provided
                if foveation_mask is not None:
                    # Downsample latents for DiT input
                    latents_input = self._downsample_latents(
                        latents_high_res_step, 
                        self.latent_height, 
                        self.latent_width,
                        foveation_mask=foveation_mask,
                        token_downsampler=token_downsampler,
                        temb=None,  # Could pass time embedding if downsampler uses it
                    )

                    # Get low-res z_t for x0 prediction
                    z_t_low_packed = self._downsample_remaining_high_res(
                        latents_input,
                        self.latent_height,
                        self.latent_width,
                        foveation_mask=foveation_mask,
                    )
                    h_2, w_2 = self.latent_height // 2, self.latent_width // 2
                    z_t_low_spatial = z_t_low_packed.view(batch_size, h_2, w_2, -1).permute(0, 3, 1, 2)
                    z_t_low_unpacked = self._unpatchify_latents(z_t_low_spatial)

                    # Prepare position IDs for foveated latents
                    latent_ids, resolution_mask, resolution_mask_top_left = self._prepare_latent_image_ids_foveation(
                        batch_size * num_images_per_prompt,
                        self.latent_height,
                        self.latent_width,
                        device,
                        latents.dtype,
                        foveation_mask,
                    )
                    if latent_ids.ndim == 2:
                        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

                    self._attention_kwargs.update({
                        "resolution_mask": resolution_mask,
                        "resolution_mask_top_left": resolution_mask_top_left,
                    })
                else:
                    latents_input = latents_high_res_step
                    z_t_low_unpacked = None

                # Forward through transformer
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=self._attention_kwargs,
                        return_dict=False,
                    )[0]

                # Upsample using x0 prediction
                if foveation_mask is not None:
                    noise_pred = self._upsample_x0_pred_unpacked(
                        noise_pred,
                        self.latent_height,
                        self.latent_width,
                        foveation_mask=foveation_mask,
                        token_upsampler=token_upsampler,
                        temb=None,
                        sigma=sigma,
                        z_t_high=z_t_high_unpacked,
                        z_t_low=z_t_low_unpacked,
                    )
                    # Pack back
                    noise_pred = self._patchify_latents(noise_pred)
                    noise_pred = self._pack_latents(noise_pred)
                else:
                    # Slice to latent size only
                    noise_pred = noise_pred[:, :latents_high_res_step.size(1), :]

                # CFG
                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_ids,
                            joint_attention_kwargs=self._attention_kwargs,
                            return_dict=False,
                        )[0]

                    if foveation_mask is not None:
                        neg_noise_pred = self._upsample_x0_pred_unpacked(
                            neg_noise_pred,
                            self.latent_height,
                            self.latent_width,
                            foveation_mask=foveation_mask,
                            token_upsampler=token_upsampler,
                            temb=None,
                            sigma=sigma,
                            z_t_high=z_t_high_unpacked,
                            z_t_low=z_t_low_unpacked,
                        )
                        neg_noise_pred = self._patchify_latents(neg_noise_pred)
                        neg_noise_pred = self._pack_latents(neg_noise_pred)
                    else:
                        neg_noise_pred = neg_noise_pred[:, :latents_high_res_step.size(1), :]

                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                # Scheduler step
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents_high_res_step, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                # Callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # 9. Decode latents
        if output_type == "latent":
            # Just return the latents without decoding
            latents_spatial = latents.view(
                batch_size, self.latent_height, self.latent_width, -1
            ).permute(0, 3, 1, 2)
            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents_spatial.device, latents_spatial.dtype)
            latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
                latents_spatial.device, latents_spatial.dtype
            )
            latents_spatial = latents_spatial * latents_bn_std + latents_bn_mean
            image = self._unpatchify_latents(latents_spatial)
        elif decode_mode == "direct":
            # Direct decoding: decode full-resolution latents
            latents_spatial = latents.view(
                batch_size, self.latent_height, self.latent_width, -1
            ).permute(0, 3, 1, 2)

            # Apply batch norm denormalization
            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents_spatial.device, latents_spatial.dtype)
            latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
                latents_spatial.device, latents_spatial.dtype
            )
            latents_spatial = latents_spatial * latents_bn_std + latents_bn_mean

            # Unpatchify
            latents_unpacked = self._unpatchify_latents(latents_spatial)
            image = self.vae.decode(latents_unpacked, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        elif decode_mode == "merge":
            # Merge decoding: decode low-res and high-res separately, then merge
            # Following exact logic from pipeline_flux_foveation_x0pred.py
            
            # Helper to apply batch norm denormalization
            def apply_bn_denorm(lat_spatial):
                bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(lat_spatial.device, lat_spatial.dtype)
                bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
                    lat_spatial.device, lat_spatial.dtype
                )
                return lat_spatial * bn_std + bn_mean

            # Step 1: Convert full-res latents to mixed-res latents
            # (HR blocks keep 4 tokens, LR blocks keep 1 token)
            mixed_res_latents = self._downsample_latents(
                latents, self.latent_height, self.latent_width, 
                foveation_mask=foveation_mask, token_downsampler=token_downsampler, temb=None
            )
            
            # Step 2: Low-res path - downsample mixed-res to uniform low-res grid
            low_res_latents = self._downsample_remaining_high_res(
                mixed_res_latents, self.latent_height, self.latent_width, foveation_mask=foveation_mask
            )
            # Convert to spatial and decode
            h_2, w_2 = self.latent_height // 2, self.latent_width // 2
            low_res_spatial = low_res_latents.view(batch_size, h_2, w_2, -1).permute(0, 3, 1, 2)
            low_res_spatial = apply_bn_denorm(low_res_spatial)
            low_res_unpacked = self._unpatchify_latents(low_res_spatial)
            low_res_image = self.vae.decode(low_res_unpacked, return_dict=False)[0]

            # Step 3: High-res path - upsample mixed-res to uniform high-res grid
            high_res_latents = self._upsample_latents(
                mixed_res_latents, self.latent_height, self.latent_width, foveation_mask=foveation_mask
            )
            high_res_spatial = high_res_latents.view(
                batch_size, self.latent_height, self.latent_width, -1
            ).permute(0, 3, 1, 2)
            high_res_spatial = apply_bn_denorm(high_res_spatial)
            high_res_unpacked = self._unpatchify_latents(high_res_spatial)
            high_res_image = self.vae.decode(high_res_unpacked, return_dict=False)[0]

            # Step 4: Upsample low-res image to high-res size
            low_res_image = F.interpolate(low_res_image, size=(height, width), mode='bilinear', align_corners=False)

            # Step 5: Create high-res foveation mask and merge images
            high_res_foveation_mask = F.interpolate(
                foveation_mask[None, None, :, :].float(), 
                size=(height, width), 
                mode='bilinear'
            ).to(latents.dtype).to(latents.device)  # [1, 1, H, W]
            
            # Merge: use high-res where foveation mask is 1, low-res elsewhere
            image = low_res_image * (1 - high_res_foveation_mask) + high_res_image * high_res_foveation_mask
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            raise ValueError(f"decode_mode '{decode_mode}' is not supported. Use 'direct' or 'merge'.")

        # Offload models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)
