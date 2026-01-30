"""
Z-Image Pipeline with Foveated Diffusion Support

This module implements the foveated diffusion pipeline that supports mixed-resolution
latent processing. Based on the FLUX foveation implementation.

Key features:
- Latent downsampling: Reduces sequence length by merging LR 2x2 blocks into single tokens
- Latent upsampling: Restores full resolution using learned or interpolation-based methods
- Resolution mask handling for CRPA attention
- Compatible with Z-Image's pipeline structure
"""

import torch
import math
from PIL import Image
from typing import Union, List, Optional, Tuple, Dict, Any, Callable
from tqdm import tqdm
from einops import rearrange
import numpy as np
import torch.nn.functional as F

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..core.data.operators import ImageCropAndResize
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput

from transformers import AutoTokenizer
from ..models.z_image_text_encoder import ZImageTextEncoder
from ..models.z_image_dit_fov import ZImageDiTFoveated
from ..models.flux_vae import FluxVAEEncoder, FluxVAEDecoder
from ..models.siglip2_image_encoder import Siglip2ImageEncoder428M


class ZImageFoveatedPipeline(BasePipeline):
    """
    Z-Image Pipeline with Foveated Diffusion Support.
    
    This pipeline extends the base Z-Image pipeline to support:
    - Mixed-resolution token sequences
    - Foveation masks for selective high-resolution regions
    - Learned or interpolation-based latent up/downsampling
    """

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("Z-Image")
        self.text_encoder: ZImageTextEncoder = None
        self.dit: ZImageDiTFoveated = None
        self.vae_encoder: FluxVAEEncoder = None
        self.vae_decoder: FluxVAEDecoder = None
        self.image_encoder: Siglip2ImageEncoder428M = None
        self.tokenizer: AutoTokenizer = None
        self.in_iteration_models = ("dit",)
        self.units = [
            ZImageFoveatedUnit_ShapeChecker(),
            ZImageFoveatedUnit_PromptEmbedder(),
            ZImageFoveatedUnit_NoiseInitializer(),
            ZImageFoveatedUnit_InputImageEmbedder(),
        ]
        self.model_fn = model_fn_z_image_foveated
        
        self.vae_scale_factor = 8

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
    ):
        pipe = ZImageFoveatedPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        pipe.text_encoder = model_pool.fetch_model("z_image_text_encoder")
        pipe.dit = model_pool.fetch_model("z_image_dit_foveated")
        pipe.vae_encoder = model_pool.fetch_model("flux_vae_encoder")
        pipe.vae_decoder = model_pool.fetch_model("flux_vae_decoder")
        pipe.image_encoder = model_pool.fetch_model("siglip_vision_model_428m")
        
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    # ========================================================================
    # Main Pipeline
    # ========================================================================

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
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
        num_inference_steps: int = 8,
        sigma_shift: float = None,
        # Foveation
        foveation_mask: torch.Tensor = None,
        foveation_start_step: int = 0,
        token_upsampler: torch.nn.Module = None,
        token_downsampler: torch.nn.Module = None,
        # VAE decode mode
        vae_decode_mode: str = "direct",  # "direct" or "merge"
        vae_merge_downsample_mode: str = "average",  # "average" or "nearest" (for merge mode)
        # Progress bar
        progress_bar_cmd=tqdm,
    ):
        """
        Generate images with optional foveated diffusion.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            cfg_scale: Classifier-free guidance scale
            input_image: Optional input image for img2img
            denoising_strength: Denoising strength for img2img
            height: Output image height
            width: Output image width
            seed: Random seed
            rand_device: Device for random generation
            num_inference_steps: Number of denoising steps
            sigma_shift: Sigma shift for scheduler
            foveation_mask: Optional mask for foveated generation [H, W] where 1=HR, 0=LR
            foveation_start_step: Step to start applying foveation
            token_upsampler: Optional learned upsampler module
            token_downsampler: Optional learned downsampler module
            vae_decode_mode: VAE decoding mode - "direct" (pass full latents) or "merge" 
                             (decode LR and HR separately, then merge images)
            vae_merge_downsample_mode: Downsampling mode for merge VAE decode - "average" (2x2 avg pool)
                             or "nearest" (top-left sampling, matches foveation exactly)
            progress_bar_cmd: Progress bar function
            
        Returns:
            Generated image
        """
        # Scheduler setup
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Calculate latent dimensions
        height, width = self.check_resize_height_width(height, width)

        # Process foveation mask
        if foveation_mask is not None:
            # Ensure mask is on correct device and has correct size
            foveation_mask = foveation_mask.to(self.device)

        # Parameters
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image,
            "denoising_strength": denoising_strength,
            "height": height,
            "width": width,
            "seed": seed,
            "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
        }
        
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Determine if using foveation for this step
            use_foveation = (
                foveation_mask is not None and 
                progress_id >= foveation_start_step
            )
            
            # Model forward
            # For Z-Image, latents stay in spatial format [B, C, H, W] (1, 16, 128, 128)
            # The foveation_mask is passed to the model for CRPA attention internally
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models,
                timestep=timestep,
                progress_id=progress_id,
                foveation_mask=foveation_mask if use_foveation else None,
            )
            
            # Scheduler step
            # Note: noise_pred and latents are both in spatial format [B, C, H, W]
            inputs_shared["latents"] = self.step(
                self.scheduler,
                progress_id=progress_id,
                noise_pred=noise_pred,
                latents=inputs_shared["latents"],
                **{k: v for k, v in inputs_shared.items() if k != "latents"}
            )

        # Decode
        self.load_models_to_device(['vae_decoder'])
        
        latents = inputs_shared["latents"]
        
        if vae_decode_mode == "direct" or foveation_mask is None:
            # Direct mode: decode full-resolution latents
            image = self.vae_decoder(latents)
            image = self.vae_output_to_image(image)
        elif vae_decode_mode == "merge":
            # Merge mode: decode LR and HR separately, then merge
            image = self._decode_merged(
                latents, 
                foveation_mask, 
                height, 
                width,
                downsample_mode=vae_merge_downsample_mode
            )
        else:
            raise ValueError(f"VAE decode mode '{vae_decode_mode}' is not supported. Use 'direct' or 'merge'.")
        
        self.load_models_to_device([])

        return image

    def _decode_merged(
        self,
        latents: torch.Tensor,
        foveation_mask: torch.Tensor,
        height: int,
        width: int,
        downsample_mode: str = "average",
    ) -> Image.Image:
        """
        Decode latents using merge mode: decode LR and HR separately, then merge.
        
        This approach follows the same 2x2 block downsampling used during foveation:
        1. Create low-res latents using 2x2 block downsampling (matching foveation patchification)
        2. Decode low-res latents through VAE
        3. Decode high-res latents through VAE
        4. Upsample low-res image and merge using foveation mask
        
        Args:
            latents: [B, C, H, W] full resolution latents (after unpatchify)
            foveation_mask: [H_latent, W_latent] foveation mask (1=HR, 0=LR)
            height: Output image height
            width: Output image width
            downsample_mode: "average" (pool 2x2 blocks) or "nearest" (top-left sampling)
            
        Returns:
            Merged PIL Image
        """
        device = latents.device
        dtype = latents.dtype
        B, C, H_latent, W_latent = latents.shape
        
        # 1. Create low-res latents by 2x2 block downsampling
        # This matches how foveation processes LR regions (keeping only 1 token per 2x2 block)
        low_res_latents = self._downsample_latents_block(
            latents, 
            foveation_mask, 
            downsample_mode=downsample_mode
        )  # [B, C, H/2, W/2]
        
        # 2. Decode low-res latents through VAE
        low_res_decoded = self.vae_decoder(low_res_latents)  # [B, C, H_pixel/2, W_pixel/2]
        
        # 3. Upsample low-res decoded image to full resolution
        low_res_upsampled = F.interpolate(
            low_res_decoded, 
            size=(height, width), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 4. Decode high-res (full) latents through VAE
        high_res_decoded = self.vae_decoder(latents)  # [B, C, H_pixel, W_pixel]
        
        # 5. Create pixel-space foveation mask
        # First expand to block level, then to pixel space
        # foveation_mask [H_latent, W_latent] -> pixel space [height, width]
        foveation_mask_pixel = F.interpolate(
            foveation_mask.float().unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='nearest'  # Use nearest to keep sharp mask boundaries
        ).to(dtype=dtype, device=device)  # [1, 1, H, W]
        
        # 6. Merge: HR regions from high_res_decoded, LR regions from low_res_upsampled
        # image = low_res * (1 - mask) + high_res * mask
        merged = low_res_upsampled * (1 - foveation_mask_pixel) + high_res_decoded * foveation_mask_pixel
        
        # 7. Convert to PIL Image
        image = self.vae_output_to_image(merged)
        
        return image

    @staticmethod
    def _downsample_latents_block(
        latents: torch.Tensor,
        foveation_mask: torch.Tensor,
        downsample_mode: str = "average",
    ) -> torch.Tensor:
        """
        Downsample latents matching the foveation block structure.
        
        During foveation:
        - Each 2x2 TOKEN block covers a 4x4 LATENT region
        - For LR regions: only 1 token is denoised, representing a 2x2 latent patch
        - After expand_foveated_tokens + unpatchify: the 2x2 patch is replicated 4 times in the 4x4 region
        
        For proper low-res VAE decoding:
        - For LR 4x4 regions: extract the original 2x2 denoised patch (top-left 2x2 of the 4x4)
        - For HR 4x4 regions: downsample all 4 patches (average or subsample)
        
        The structure after unpatchify for an LR 4x4 block:
            [A, B, A, B]
            [C, D, C, D]  where [A,B;C,D] is the denoised 2x2 patch
            [A, B, A, B]
            [C, D, C, D]
        We extract [A,B;C,D] from positions [0:2, 0:2].
        
        Args:
            latents: [B, C, H, W] spatial latents after unpatchify
            foveation_mask: [H, W] foveation mask (1=HR, 0=LR)
            downsample_mode: "average" or "nearest" for HR regions
            
        Returns:
            low_res_latents: [B, C, H/2, W/2]
        """
        B, C, H, W = latents.shape
        device = latents.device
        dtype = latents.dtype
        
        # Foveation operates on 2x2 TOKEN blocks = 4x4 LATENT blocks
        # Token space: H_t = H/2, W_t = W/2
        # Token block space: h_2 = H_t/2 = H/4, w_2 = W_t/2 = W/4
        H_t, W_t = H // 2, W // 2  # Token dimensions (output size)
        h_2, w_2 = H_t // 2, W_t // 2  # Token block dimensions
        
        # Process foveation mask to token block level
        # foveation_mask is [H, W], resize to token space [H_t, W_t] if needed
        mask = foveation_mask.to(device)
        if mask.shape != (H_t, W_t):
            mask = F.interpolate(
                mask.float().unsqueeze(0).unsqueeze(0),
                size=(H_t, W_t),
                mode='nearest'
            ).squeeze()
        
        # Group tokens into 2x2 blocks to determine LR vs HR
        # mask: [H_t, W_t] -> mask_blocks: [h_2, w_2, 4]
        mask_blocks = mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
        is_high_res_block = (mask_blocks.sum(dim=-1) > 0)  # [h_2, w_2]
        is_low_res_block = ~is_high_res_block
        
        # ================================================================
        # Step 1: Create LR output by taking top-left 2x2 of each 4x4 block
        # This extracts [A,B;C,D] from each 4x4 region
        # ================================================================
        # latents: [B, C, H, W] where H = 4*h_2, W = 4*w_2
        # Reshape to [B, C, h_2, 4, w_2, 4] to access 4x4 blocks
        latents_blocked = latents.view(B, C, h_2, 4, w_2, 4)
        
        # Extract top-left 2x2 from each 4x4 block: positions [0:2, 0:2]
        # lr_output: [B, C, h_2, 2, w_2, 2]
        lr_output = latents_blocked[:, :, :, 0:2, :, 0:2]
        
        # ================================================================
        # Step 2: Create HR output by downsampling each 4x4 block to 2x2
        # ================================================================
        if downsample_mode == "average":
            # For HR: average each 2x2 sub-block within the 4x4 block
            # Reshape [B, C, h_2, 4, w_2, 4] -> [B, C, h_2, 2, 2, w_2, 2, 2]
            hr_4x4 = latents_blocked.view(B, C, h_2, 2, 2, w_2, 2, 2)
            # Average the 2x2 sub-blocks (dims 4 and 7)
            hr_output = hr_4x4.mean(dim=(4, 7))  # [B, C, h_2, 2, w_2, 2]
        elif downsample_mode == "nearest":
            # For HR: take top-left of each 2x2 sub-block
            # This is positions [0, 2] in both h and w dimensions of the 4x4 block
            hr_output = latents_blocked[:, :, :, 0::2, :, 0::2]  # [B, C, h_2, 2, w_2, 2]
        else:
            raise ValueError(f"Unknown downsample_mode: {downsample_mode}")
        
        # ================================================================
        # Step 3: Combine using mask - LR regions use lr_output, HR regions use hr_output
        # ================================================================
        # Expand mask to match block dimensions: [h_2, w_2] -> [1, 1, h_2, 1, w_2, 1]
        lr_mask = is_low_res_block.view(1, 1, h_2, 1, w_2, 1).expand(B, C, h_2, 2, w_2, 2)
        
        # Select: LR blocks get lr_output, HR blocks get hr_output
        output_blocked = torch.where(lr_mask, lr_output, hr_output)
        
        # ================================================================
        # Step 4: Reshape to final output [B, C, H_t, W_t]
        # ================================================================
        # [B, C, h_2, 2, w_2, 2] -> [B, C, H_t, W_t]
        # Direct reshape works because:
        # - input[i, j, k, l] at flat pos: i*4*w_2 + j*2*w_2 + k*2 + l
        # - output[2i+j, 2k+l] at flat pos: (2i+j)*2*w_2 + (2k+l) = 4*i*w_2 + 2*j*w_2 + 2*k + l ✓
        low_res_latents = output_blocked.contiguous().view(B, C, H_t, W_t)
        
        return low_res_latents


# ========================================================================
# Pipeline Units
# ========================================================================

class ZImageFoveatedUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: ZImageFoveatedPipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class ZImageFoveatedUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params=(),
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_embeds",),
            onload_model_names=("text_encoder",)
        )

    def encode_prompt(
        self,
        pipe,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
    ) -> List[torch.FloatTensor]:
        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [{"role": "user", "content": prompt_item}]
            prompt_item = pipe.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = pipe.tokenizer(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = pipe.text_encoder(
            input_ids=text_input_ids, attention_mask=prompt_masks, output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []
        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

    def process(self, pipe: ZImageFoveatedPipeline, prompt):
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds = self.encode_prompt(pipe, prompt, pipe.device)
        return {"prompt_embeds": prompt_embeds}


class ZImageFoveatedUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: ZImageFoveatedPipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise(
            (1, 16, height // 8, width // 8),
            seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype
        )
        return {"noise": noise}


class ZImageFoveatedUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae_encoder",)
        )

    def process(self, pipe: ZImageFoveatedPipeline, input_image, noise):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae_encoder'])
        image = pipe.preprocess_image(input_image)
        input_latents = pipe.vae_encoder(image)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}


# ========================================================================
# Model Function
# ========================================================================

def model_fn_z_image_foveated(
    dit: ZImageDiTFoveated,
    latents=None,
    timestep=None,
    prompt_embeds=None,
    foveation_mask=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    """
    Model function for foveated Z-Image generation.
    
    Args:
        dit: ZImageDiTFoveated model
        latents: Input latents in spatial format [B, C, H, W]
        timestep: Current timestep (scheduler timestep, not scaled)
        prompt_embeds: Text embeddings
        foveation_mask: Foveation mask for mixed-resolution [H, W]
        use_gradient_checkpointing: Enable gradient checkpointing
        use_gradient_checkpointing_offload: Enable with offload
        
    Returns:
        Noise prediction in spatial format [B, C, H, W]
    """
    while isinstance(prompt_embeds, list):
        prompt_embeds = prompt_embeds[0]
    while isinstance(latents, list):
        latents = latents[0]

    # Ensure latents are in [B, C, H, W] format
    if latents.ndim != 4:
        raise ValueError(f"Expected latents to be 4D [B, C, H, W], got {latents.ndim}D")

    # Convert to DiT expected format: [C, B, H, W]. [1, 16, 128, 128] for a 1024x1024 image.
    latents = rearrange(latents, "B C H W -> C B H W")

    # Timestep conversion (Z-Image uses 1000 - t)
    # The DiT.forward() expects t and does t * t_scale internally for embeddings
    timestep_converted = 1000 - timestep

    # Forward through DiT
    # The DiT handles patchification and foveation internally
    model_output = dit(
        [latents],  # List of image tensors [C, B, H, W]
        timestep_converted / dit.t_scale,  # Normalized timestep
        [prompt_embeds],  # List of caption features
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        foveation_mask=foveation_mask,
    )[0]

    # Negate output (Z-Image convention)
    model_output = -model_output
    
    # Convert back to [B, C, H, W]
    if model_output.ndim == 4:
        model_output = rearrange(model_output, "C B H W -> B C H W")

    return model_output
