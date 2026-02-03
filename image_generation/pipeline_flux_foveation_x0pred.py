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
Pipeline for Foveated Diffusion with X0 (Clean Image) Prediction for Upsampling.

Key difference from pipeline_flux_foveation_tokenAE.py:
- Instead of upsampling velocity/noise predictions directly, this pipeline:
  1. Converts low-res velocity predictions to clean image predictions: x_0 = z_t - sigma * v
  2. Upsamples the clean image predictions (which have more structure)
  3. Converts back to velocity for the scheduler: v = (z_t - x_0) / sigma

This makes the upsampling task easier because clean images have semantic structure
that CNNs can learn to super-resolve, unlike noise which is unstructured.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

from transformer_flux_crpa import FluxPhaseAlignedTransformer2DModel

# utils
import sys
sys.path.append("..")
from utils import save_image

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FluxPipeline_Foveation_X0Pred(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
    FluxIPAdapterMixin,
):
    r"""
    The Flux pipeline for text-to-image generation with foveated rendering and X0 prediction upsampling.

    This variant uses clean image (x0) prediction for upsampling low-resolution regions,
    which is easier for the upsampler network to learn compared to upsampling noise directly.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxPhaseAlignedTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

    
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return image_embeds

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt
    ):
        image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != self.transformer.encoder_hid_proj.num_ip_adapters:
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_ip_adapter_image in ip_adapter_image:
                single_image_embeds = self.encode_image(single_ip_adapter_image, device, 1)
                image_embeds.append(single_image_embeds[None, :])
        else:
            if not isinstance(ip_adapter_image_embeds, list):
                ip_adapter_image_embeds = [ip_adapter_image_embeds]

            if len(ip_adapter_image_embeds) != self.transformer.encoder_hid_proj.num_ip_adapters:
                raise ValueError(
                    f"`ip_adapter_image_embeds` must have same length as the number of IP Adapters. Got {len(ip_adapter_image_embeds)} image embeds and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_image_embeds in ip_adapter_image_embeds:
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for single_image_embeds in image_embeds:
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    
    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype, foveation_mask=None):
        # latent_image_ids [height, width, 3]
        latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        # 4th channel for resolution mask (1=HR, 0=LR)
        # We create a temporary mask tensor
        resolution_mask_grid = torch.ones(height, width, device=device, dtype=dtype) 

        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height, device=device, dtype=dtype)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width, device=device, dtype=dtype)[None, :]

        if foveation_mask is not None:
            if height % 2 != 0 or width % 2 != 0:
                raise ValueError("Height and width must be even for mixed-resolution stride 2.")
            h_2, w_2 = height // 2, width // 2

            # Reshape to blocks
            grid_blocks = latent_image_ids.view(h_2, 2, w_2, 2, 3).permute(0, 2, 1, 3, 4).reshape(h_2, w_2, 4, 3) # [32, 32, 4, 3]
            
            # Mask blocks
            mask_blocks = foveation_mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4) # [32, 32, 4]
            is_high_res_block = (mask_blocks.sum(dim=-1) > 0) # [32, 32]
            is_low_res_block = ~is_high_res_block # [32, 32]

            # Handle IDs for RoPE (Averaging for LR)
            mode = "top_left"
            if mode == "average":
                low_res_coords = grid_blocks.mean(dim=2)
            elif mode == "top_left":
                low_res_coords = grid_blocks[..., 0, :] # [32, 32, 3]
            else:
                raise ValueError(f"Unknown mode '{mode}'. Use 'average' or 'top_left'.")
            output_grid = grid_blocks.clone()
            output_grid[is_low_res_block, 0, :] = low_res_coords[is_low_res_block]

            # Handle Resolution Mask
            # Reshape resolution grid to blocks [H/2, W/2, 4]
            res_mask_blocks = resolution_mask_grid.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4) # [32, 32, 4]
            output_res_mask = res_mask_blocks.clone()
            
            # Set LR blocks to 0
            output_res_mask[is_low_res_block, :] = 0.0 

            # Filter Validity
            valid_tokens = torch.ones(h_2, w_2, 4, device=device, dtype=torch.bool)
            valid_tokens[is_low_res_block, 1:] = False

            output_grid = output_grid.reshape(-1, 3)
            output_res_mask = output_res_mask.reshape(-1)
            valid_tokens = valid_tokens.reshape(-1)

            # top-left high-res mask for subsampling
            tl_mask_blocks = torch.zeros_like(res_mask_blocks)
            tl_mask_blocks[..., 0] = 1.0 # top-left high-res mask 
            tl_mask_blocks[is_low_res_block, :] = 1.0 # low-res blocks
            resolution_mask_top_left = tl_mask_blocks.reshape(-1)[valid_tokens.reshape(-1)]

            latent_image_ids = output_grid[valid_tokens] # [Seq_Len, 3]
            resolution_mask = output_res_mask[valid_tokens] # [Seq_Len], True for HR, False for LR
            # find the number of tokens with fractional strides in x or Y - IF strides 
            fractional_strides = (output_grid[..., 1] % 1 != 0) | (output_grid[..., 2] % 1 != 0)
            num_fractional_strides = fractional_strides.sum()
            
            num_hr_tokens = resolution_mask.sum().item()
            num_lr_tokens = resolution_mask.shape[0] - num_hr_tokens
            num_hr_top_left_tokens = num_hr_tokens // 4

            return latent_image_ids.to(device=device, dtype=dtype), resolution_mask.to(device=device, dtype=dtype), resolution_mask_top_left.to(device=device, dtype=dtype)
        
        else:
            # Standard Path
            latent_image_ids = latent_image_ids.reshape(-1, 3)
            resolution_mask = resolution_mask_grid.reshape(-1) # dummy, not used
            resolution_mask_top_left = resolution_mask_grid.clone().reshape(-1) # dummy, not used
            return latent_image_ids.to(device=device, dtype=dtype), None, None
    
    @staticmethod
    def _downsample_latents(latents, height, width, foveation_mask=None, sample_mode="top_left", token_downsampler=None, temb=None):
        """
        Downsamples latents spatially based on the foveation mask to match the 
        mixed-resolution structure of latent_image_ids.
        
        Args:
            latents: [Batch, Seq_Len, Channels]
            height: Height of the latent grid (e.g. original_height // 2)
            width: Width of the latent grid (e.g. original_width // 2)
            foveation_mask: [height, width] Tensor indicating high-res regions for packed latents
            sample_mode: "average" to mean-pool 2x2 blocks, "top_left" to keep the top-left pixel.
        """
        if foveation_mask is None:
            return latents
        mask = foveation_mask.to(latents.device)

        batch_size, seq_len, channels = latents.shape
        
        # Verify dimensions
        if height * width != seq_len:
            raise ValueError(f"Height ({height}) * Width ({width}) does not match sequence length ({seq_len}).")

        output_latents = latents.clone() # [B, 64, 64, 64] # B, H, W, C -> high-res latents should be in [32, 2, 32, 2] order

        # downsample latents
        # unpack latents back to original spatial order 
        latents = latents.view(batch_size, height, width, channels // 4, 2, 2) # [B, 64, 64, 16, 2, 2], 0, 1, 2, 3, 4, 5
        latents = latents.permute(0, 3, 1, 4, 2, 5) # [B, 16, 64, 2, 64, 2]
        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2) # [B, 16, 128, 128] # original spatial order

        # make sure downsampling is done here
        sample_mode = "average"
        if token_downsampler is not None:
            latents = token_downsampler(latents, temb=temb, foveation_mask=foveation_mask).to(output_latents.dtype)
        elif sample_mode == "average":
            latents = F.interpolate(latents, scale_factor=0.5, mode="bilinear") * 2 # [B, 16, 64, 64]
        elif sample_mode == "top_left":
            latents = F.interpolate(latents, scale_factor=0.5, mode="nearest")
        else:
            raise ValueError(f"Unknown sample_mode '{sample_mode}'. Use 'average' or 'top_left'.")
        latents = latents.reshape(batch_size, channels // (2 * 2), height // 2, 2, width // 2, 2) # [B, 16, 32, 2, 32, 2], (0, 1, 2, 3, 4, 5)
        latents = latents.permute(0, 2, 4, 1, 3, 5) # [B, 32, 32, 16, 2, 2]
        low_res_packed_latents = latents.reshape(batch_size, height // 2, width // 2, channels) # [B, 32, 32, 64]


        h_2, w_2 = height // 2, width // 2
        output_latents = output_latents.view(batch_size, height, width, channels)
        output_latents = output_latents.view(batch_size, h_2, 2, w_2, 2, channels) # output latents as blocks
        # The reshape puts the 2x2 block elements in order: (0,0), (0,1), (1,0), (1,1)
        # So index 0 is Top-Left.
        output_latents = output_latents.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, h_2, w_2, 4, channels) # [B, 32, 32, 4, 64]

        # 2. Reshape Mask into 2x2 blocks
        # [H, W] -> [H/2, W/2, 4]
        mask_blocks = mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)

        # 3. Determine Block Status
        # High Res = Any 1 in the 2x2 block
        # Low Res = All 0 in the 2x2 block
        is_high_res_block = (mask_blocks.sum(dim=-1) > 0) # [H/2, W/2]
        is_low_res_block = ~is_high_res_block

        output_latents[:, is_low_res_block, 0, :] = low_res_packed_latents[:, is_low_res_block, :]

        valid_tokens = torch.ones(h_2, w_2, 4, device=latents.device, dtype=torch.bool)
        
        # Mark indices 1, 2, 3 as invalid for low-res blocks.
        # This keeps only index 0 (which is either the Average or the Top-Left pixel depending on mode).
        valid_tokens[is_low_res_block, 1:] = False

        # Flatten and Select
        output_latents = output_latents.view(batch_size, -1, channels) # [B, Total_Tokens, C]
        valid_tokens = valid_tokens.view(-1) # [Total_Tokens]
        latents_downsampled = output_latents[:, valid_tokens, :]
        # latents in [32, 32, 4] order
        return latents_downsampled

    @staticmethod
    def _downsample_remaining_high_res(latents, height, width, foveation_mask, downsample_mode="average"):
        """
        Takes mixed-resolution latents (where some blocks are 1 token and others are 4)
        and downsamples the high-resolution blocks so that ALL blocks become 1 token.
        
        Args:
            latents: [Batch, Reduced_Seq_Len, Channels] (The mixed-res input)
            height: Original grid height.
            width: Original grid width.
            foveation_mask: [height, width] Tensor used to define the CURRENT high-res regions.
            downsample_mode: "average" (pools 4 tokens -> 1) or "nearest" (keeps top-left).
            
        Returns:
            uniform_latents: [Batch, (H/2 * W/2), Channels]
        """
        batch_size, reduced_len, channels = latents.shape
        device = latents.device
        dtype = latents.dtype
        
        h_2, w_2 = height // 2, width // 2
        
        # 1. Determine Block Status (Same logic as upsampler)
        mask = foveation_mask.to(device)
        # [H, W] -> [H/2, W/2, 4]
        mask_blocks = mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
        
        # Determine which blocks currently have 4 tokens (High Res)
        is_high_res_block = (mask_blocks.sum(dim=-1) > 0) # [H/2, W/2]
        is_low_res_block = ~is_high_res_block
        
        # 2. Reconstruct the Block Structure
        # We need to map the sparse 1D sequence back to the 2D block grid
        
        # Create Layout Mask: Which indices in the 2x2 blocks are active in the input?
        # Low Res blocks have index 0 active. High Res have 0,1,2,3 active.
        layout_mask = torch.ones(h_2, w_2, 4, device=device, dtype=torch.bool)
        layout_mask[is_low_res_block, 1:] = False
        
        if layout_mask.sum() != reduced_len:
             raise ValueError(f"Latent length {reduced_len} does not match mask topology {layout_mask.sum()}.")

        # Scatter into a dense buffer: [B, H/2, W/2, 4, C]
        # We initialize with zeros, but strictly we only care about the slots defined by layout_mask.
        reconstructed_blocks = torch.zeros(batch_size, h_2 * w_2 * 4, channels, device=device, dtype=dtype)
        reconstructed_blocks[:, layout_mask.view(-1), :] = latents
        reconstructed_blocks = reconstructed_blocks.view(batch_size, h_2, w_2, 4, channels)

        # 3. Perform Downsampling on High-Res Blocks
        # We want every block to end up with a single valid value at index 0.
        
        # For Low-Res blocks: The value is already at index 0. We do nothing.
        
        # For High-Res blocks: We currently have values at 0,1,2,3. We need to pool them into index 0.
        if is_high_res_block.any():
            high_res_tokens = reconstructed_blocks[:, is_high_res_block, :, :] # [B, N_High, 4, C]
            
            if downsample_mode == "average":
                # Average the 4 tokens
                pooled_val = high_res_tokens.mean(dim=2) # [B, N_High, C]
                reconstructed_blocks[:, is_high_res_block, 0, :] = pooled_val
            elif downsample_mode == "nearest":
                # Keep index 0 (top-left), discard 1,2,3.
                # Since index 0 is already there, we technically do nothing.
                pass
        
        # 4. Extract Uniform Grid
        # Now every block (whether originally High or Low res) has its representative value at index 0.
        # We discard indices 1, 2, 3 for everyone.
        
        final_grid = reconstructed_blocks[:, :, :, 0, :] # [B, H/2, W/2, C]
        
        # Flatten to sequence
        return final_grid.reshape(batch_size, h_2 * w_2, channels)

    @staticmethod
    def _upsample_x0_pred_unpacked(
        noise_pred, 
        latent_height, 
        latent_width, 
        vae_scale_factor, 
        foveation_mask=None, 
        token_upsampler=None, 
        temb=None,
        sigma=None,
        z_t_high=None,
        z_t_low=None,
    ):
        """
        Upsamples using X0 (clean image) prediction domain instead of noise domain.
        
        This is easier for the upsampler to learn because clean images have semantic
        structure that CNNs can exploit, unlike noise which is unstructured.
        
        Algorithm:
        1. For low-res regions:
           - Get low-res velocity v_low from DiT (via _downsample_remaining_high_res)
           - Use z_t_low (pre-computed using same downsampling logic)
           - Compute x_0_low = z_t_low - sigma * v_low (clean image prediction)
           - Upsample x_0_low to x_0_high using the learned upsampler
           - Convert back to velocity: v_high = (z_t_high - x_0_high) / sigma
        2. For high-res regions:
           - Keep original DiT predictions (already high-res)
        
        Args:
            noise_pred: [B, Reduced_Seq_Len, 64] - Mixed resolution velocity predictions from DiT
            latent_height: Height of the packed latent grid (e.g., 64 for 1024px images)
            latent_width: Width of the packed latent grid
            vae_scale_factor: VAE compression factor (typically 8)
            foveation_mask: [latent_height, latent_width] - Binary mask for high-res regions
            token_upsampler: Learned upsampler network
            temb: Time embedding for the upsampler
            sigma: [B, 1, 1, 1] - Noise level at current timestep for x0 conversion
            z_t_high: [B, 16, H_pixel, W_pixel] - High-res noisy latents (unpacked)
            z_t_low: [B, 16, H_pixel/2, W_pixel/2] - Low-res noisy latents (unpacked), 
                     computed using _downsample_latents -> _downsample_remaining_high_res -> _unpack_latents
            
        Returns:
            [B, 16, H_pixel, W_pixel] - Full resolution velocity prediction (unpacked)
        """
        if foveation_mask is None:
            noise_pred_unpacked = FluxPipeline_Foveation_X0Pred._unpack_latents(
                noise_pred,
                height=latent_height * 2 * vae_scale_factor,
                width=latent_width * 2 * vae_scale_factor,
                vae_scale_factor=vae_scale_factor
            )
            return noise_pred_unpacked

        batch_size = noise_pred.shape[0]
        device = noise_pred.device
        dtype = noise_pred.dtype
        
        pixel_height = latent_height * 2
        pixel_width = latent_width * 2

        # =====================================================================
        # LOW-RES REGIONS: X0 Prediction Upsampling
        # =====================================================================
        
        # 1. Get uniform low-res velocity by averaging high-res predictions
        # This creates a [B, H/2 * W/2, 64] grid where all blocks are single tokens
        v_low_packed = FluxPipeline_Foveation_X0Pred._downsample_remaining_high_res(
            noise_pred, 
            latent_height, 
            latent_width, 
            foveation_mask=foveation_mask
        )
        
        # 2. Unpack to spatial format: [B, 16, H/2, W/2]
        v_low_unpacked = FluxPipeline_Foveation_X0Pred._unpack_latents(
            v_low_packed, 
            height=latent_height * vae_scale_factor,  # H/2 in pixel space
            width=latent_width * vae_scale_factor,    # W/2 in pixel space
            vae_scale_factor=vae_scale_factor
        )  # [B, 16, H/2, W/2] e.g., [B, 16, 64, 64]
        
        # 3. z_t_low is passed in (computed using same downsampling logic as v_low:
        #    _downsample_latents -> _downsample_remaining_high_res -> _unpack_latents)
        # z_t_low: [B, 16, H/2, W/2], z_t_high: [B, 16, H, W]
        
        # 4. Convert velocity to clean image prediction: x_0 = z_t - sigma * v
        # sigma shape should be [B, 1, 1, 1] for broadcasting
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        elif sigma.dim() == 2:
            sigma = sigma.view(sigma.shape[0], 1, 1, 1)
            
        x_0_low = z_t_low - sigma * v_low_unpacked  # [B, 16, H/2, W/2]
        
        # 5. Upsample clean image prediction (this is what the upsampler learns!)
        x_0_high_upsampled = token_upsampler(x_0_low, temb=temb, foveation_mask=foveation_mask)  # [B, 16, H, W]
        
        # 6. Convert back to velocity: v = (z_t - x_0) / sigma
        # Clamp sigma to avoid division by zero at the end of denoising
        sigma_clamped = sigma.clamp(min=1e-6)
        v_high_from_upsampler = (z_t_high - x_0_high_upsampled) / sigma_clamped  # [B, 16, H, W]

        # =====================================================================
        # HIGH-RES REGIONS: Keep Original DiT Predictions
        # =====================================================================
        
        h_2, w_2 = latent_height // 2, latent_width // 2
        mask = foveation_mask
        mask_blocks = mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
        is_high_res_block = (mask_blocks.sum(dim=-1) > 0) 
        is_low_res_block = ~is_high_res_block

        # Map sparse sequence to dense blocks
        layout_mask = torch.ones(h_2, w_2, 4, device=device, dtype=torch.bool)
        layout_mask[is_low_res_block, 1:] = False  # Low res only uses index 0

        reconstructed_blocks = torch.zeros(
            batch_size, h_2 * w_2 * 4, noise_pred.shape[-1], 
            device=device, dtype=dtype
        )
        reconstructed_blocks[:, layout_mask.view(-1), :] = noise_pred
        reconstructed_blocks = reconstructed_blocks.view(batch_size, h_2, w_2, 4, -1)

        # Fill Low-Res Holes with nearest neighbor (just for unpacking structure)
        if is_low_res_block.any():
            lr_tokens = reconstructed_blocks[:, is_low_res_block, 0:1, :]
            reconstructed_blocks[:, is_low_res_block, 1:, :] = lr_tokens

        # View as Spatial Grid of Patches [B, H, W, 64]
        dense_packed_grid = reconstructed_blocks.view(
            batch_size, h_2, w_2, 2, 2, -1
        ).permute(0, 1, 3, 2, 4, 5).reshape(batch_size, latent_height, latent_width, -1)

        # Reshape to expected input for _unpack_latents: [B, H*W, 64]
        flat_packed = dense_packed_grid.reshape(batch_size, -1, 64)
        
        # Unpack high-res predictions
        v_high_original = FluxPipeline_Foveation_X0Pred._unpack_latents(
            flat_packed, 
            height=pixel_height * vae_scale_factor,
            width=pixel_width * vae_scale_factor,
            vae_scale_factor=vae_scale_factor
        )  # [B, 16, H, W]

        # =====================================================================
        # COMBINE: Use mask to select HR vs LR regions
        # =====================================================================
        
        # Upsample foveation mask to pixel resolution
        foveation_mask_upsampled = F.interpolate(
            foveation_mask.float().unsqueeze(0).unsqueeze(0), 
            size=(pixel_height, pixel_width), 
            mode='nearest'
        ).squeeze().bool()
        
        # Final output: HR regions use original DiT predictions, LR regions use x0-upsampled
        final_output = torch.where(
            foveation_mask_upsampled.to(device), 
            v_high_original.to(device),           # High-res regions: keep original
            v_high_from_upsampler.to(device)      # Low-res regions: use x0-upsampled
        )
        
        return final_output

    @staticmethod
    def _upsample_latents(latents, height, width, foveation_mask=None, upsample_mode="bilinear"):
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
        
        is_high_res_block = (mask_blocks.sum(dim=-1) > 0) # [H/2, W/2]
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
            coarse_grid = reconstructed_blocks[..., 0, :].clone() # [B, H/2, W/2, C]
            
            # If block is high-res, replace with mean of all 4 tokens to prevent artifacts
            if is_high_res_block.any():
                high_res_means = reconstructed_blocks[:, is_high_res_block, :, :].mean(dim=2)
                coarse_grid[:, is_high_res_block, :] = high_res_means
            
            # Rearrange for interpolation: [B, C, H/2, W/2]
            coarse_grid = coarse_grid.permute(0, 3, 1, 2)
            
            # B. Interpolate to Full Res [B, C, H, W]
            fine_grid = torch.nn.functional.interpolate(
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
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        depr_message = f"Calling `enable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_slicing()`."
        deprecate(
            "enable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_slicing()`."
        deprecate(
            "disable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        depr_message = f"Calling `enable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_tiling()`."
        deprecate(
            "enable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_tiling()`."
        deprecate(
            "disable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_tiling()

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None, foveation_mask=None):
        # ... [Keep existing sizing logic] ...
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        self.latent_height = height // 2
        self.latent_width = width // 2
        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
             # Just get IDs and Mask
            latent_image_ids, resolution_mask, resolution_mask_top_left = self._prepare_latent_image_ids(batch_size, self.latent_height, self.latent_width, device, dtype, foveation_mask=foveation_mask)
            return latents.to(device=device, dtype=dtype), latent_image_ids, resolution_mask, resolution_mask_top_left

        # ... [Generation logic] ...
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        
        latent_image_ids, resolution_mask, resolution_mask_top_left = self._prepare_latent_image_ids(batch_size, self.latent_height, self.latent_width, device, dtype, foveation_mask=foveation_mask)

        return latents, latent_image_ids, resolution_mask, resolution_mask_top_left

    def save_intermediate_x0(
        self,
        noise_pred,
        z_t_high_unpacked,
        z_t_low_unpacked,
        sigma,
        foveation_mask,
        token_upsampler,
        temb,
        step_idx,
        save_dir,
        height,
        width,
    ):
        """
        Save intermediate x0 (clean image) predictions and velocity maps at a denoising step.
        
        This function computes and saves:
        
        X0 (Clean Image) Predictions (decoded through VAE):
        1. x0_low: Low-res clean image prediction (uniform low-res grid)
        2. x0_high_upsampled: High-res clean image from upsampling x0_low
        3. x0_high_original: High-res clean image from DiT's HR predictions (for HR regions)
        4. x0_merged: Final merged x0 (HR regions from DiT, LR regions from upsampler)
        
        Velocity Predictions (color-coded maps):
        For each velocity, saves 3 visualizations:
        - *_mean: Mean across channels with RdBu diverging colormap (red=positive, blue=negative)
        - *_magnitude: L2 norm across channels with hot colormap
        - *_rgb: First 3 channels mapped to RGB
        
        Velocity types saved:
        5. v_low: Low-res velocity from DiT
        6. v_high_original: High-res velocity from DiT (with NN-filled LR regions)
        7. v_high_from_upsampler: Velocity computed from upsampled x0
        8. v_merged: Final merged velocity (HR from DiT, LR from upsampler)
        
        X0 Latent Visualizations (color-coded maps, same format as velocity):
        9. x0_low_latent: Low-res x0 latent visualization
        10. x0_high_upsampled_latent: Upsampled x0 latent visualization
        11. x0_high_original_latent: Original high-res x0 latent visualization
        12. x0_merged_latent: Merged x0 latent visualization
        
        Spatial Gradient Maps (shows local noisiness/detail):
        For each tensor, saves 4 visualizations:
        - *_grad_mean: Mean gradient magnitude across channels (hot colormap)
        - *_grad_l2: L2 norm of gradient magnitude (hot colormap)
        - *_noisiness: Same as grad_l2 with inferno colormap (dark=smooth, bright=noisy)
        - *_grad_direction: Gradient direction/angle (hsv colormap, shows edge orientation)
        
        Saved for: x0_low, x0_high_upsampled, x0_high_original, x0_merged,
                   v_low, v_high_original, v_high_from_upsampler, v_merged,
                   z_t_high, z_t_low
        
        Args:
            noise_pred: [B, Reduced_Seq_Len, 64] - Mixed resolution velocity predictions from DiT
            z_t_high_unpacked: [B, 16, H, W] - High-res noisy latents (unpacked)
            z_t_low_unpacked: [B, 16, H/2, W/2] - Low-res noisy latents (unpacked)
            sigma: [B, 1, 1, 1] - Noise level at current timestep
            foveation_mask: [latent_height, latent_width] - Binary mask for high-res regions
            token_upsampler: Learned upsampler network
            temb: Time embedding
            step_idx: Current denoising step index
            save_dir: Directory to save images
            height: Output image height
            width: Output image width
        """
        import os
        from pathlib import Path
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        batch_size = noise_pred.shape[0]
        device = noise_pred.device
        dtype = noise_pred.dtype
        
        latent_height = self.latent_height
        latent_width = self.latent_width
        pixel_height = latent_height * 2
        pixel_width = latent_width * 2
        
        # Ensure sigma has correct shape
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        elif sigma.dim() == 2:
            sigma = sigma.view(sigma.shape[0], 1, 1, 1)
        
        with torch.no_grad():
            # =====================================================================
            # 1. Compute x0_low from v_low and z_t_low
            # =====================================================================
            # Get uniform low-res velocity
            v_low_packed = self._downsample_remaining_high_res(
                noise_pred, 
                latent_height, 
                latent_width, 
                foveation_mask=foveation_mask
            )
            v_low_unpacked = self._unpack_latents(
                v_low_packed, 
                height=latent_height * self.vae_scale_factor,
                width=latent_width * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor
            )  # [B, 16, H/2, W/2]
            
            # x0_low = z_t_low - sigma * v_low
            x_0_low = z_t_low_unpacked - sigma * v_low_unpacked  # [B, 16, H/2, W/2]
            
            # =====================================================================
            # 2. Compute x0_high_upsampled by upsampling x0_low
            # =====================================================================
            x_0_high_upsampled = token_upsampler(x_0_low, temb=temb, foveation_mask=foveation_mask)  # [B, 16, H, W]
            
            # =====================================================================
            # 3. Compute x0_high_original from DiT's high-res predictions
            # =====================================================================
            h_2, w_2 = latent_height // 2, latent_width // 2
            mask = foveation_mask
            mask_blocks = mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
            is_high_res_block = (mask_blocks.sum(dim=-1) > 0) 
            is_low_res_block = ~is_high_res_block
            
            # Map sparse sequence to dense blocks
            layout_mask = torch.ones(h_2, w_2, 4, device=device, dtype=torch.bool)
            layout_mask[is_low_res_block, 1:] = False
            
            reconstructed_blocks = torch.zeros(
                batch_size, h_2 * w_2 * 4, noise_pred.shape[-1], 
                device=device, dtype=dtype
            )
            reconstructed_blocks[:, layout_mask.view(-1), :] = noise_pred
            reconstructed_blocks = reconstructed_blocks.view(batch_size, h_2, w_2, 4, -1)
            
            # Fill Low-Res Holes with nearest neighbor
            if is_low_res_block.any():
                lr_tokens = reconstructed_blocks[:, is_low_res_block, 0:1, :]
                reconstructed_blocks[:, is_low_res_block, 1:, :] = lr_tokens
            
            dense_packed_grid = reconstructed_blocks.view(
                batch_size, h_2, w_2, 2, 2, -1
            ).permute(0, 1, 3, 2, 4, 5).reshape(batch_size, latent_height, latent_width, -1)
            
            flat_packed = dense_packed_grid.reshape(batch_size, -1, 64)
            
            # Unpack to get v_high_original
            v_high_original = self._unpack_latents(
                flat_packed, 
                height=pixel_height * self.vae_scale_factor,
                width=pixel_width * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor
            )  # [B, 16, H, W]
            
            # x0_high_original = z_t_high - sigma * v_high_original
            x_0_high_original = z_t_high_unpacked - sigma * v_high_original  # [B, 16, H, W]
            
            # =====================================================================
            # 4. Compute merged x0 (HR regions from original, LR from upsampled)
            # =====================================================================
            foveation_mask_upsampled = F.interpolate(
                foveation_mask.float().unsqueeze(0).unsqueeze(0), 
                size=(pixel_height, pixel_width), 
                mode='nearest'
            ).squeeze().bool()
            
            x_0_merged = torch.where(
                foveation_mask_upsampled.to(device), 
                x_0_high_original,      # High-res regions: use original DiT prediction
                x_0_high_upsampled      # Low-res regions: use upsampled
            )
            
            # =====================================================================
            # 5. Decode through VAE and save images
            # =====================================================================
            def decode_and_save(latents_to_decode, name, is_low_res=False):
                """Helper to decode latents through VAE and save."""
                # Apply VAE scaling
                latents_scaled = (latents_to_decode / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                
                # Decode
                decoded = self.vae.decode(latents_scaled.to(self.vae.dtype), return_dict=False)[0]
                
                # Post-process
                images = self.image_processor.postprocess(decoded, output_type="pil")
                
                # Save each image in batch
                for b_idx, img in enumerate(images):
                    if is_low_res:
                        # Upsample low-res image to match high-res size for comparison
                        img = img.resize((width, height), resample=Image.BILINEAR)
                    save_path = save_dir / f"step_{step_idx:03d}_{name}_batch_{b_idx}.png"
                    img.save(save_path)
            
            from PIL import Image
            
            # Save x0_low (low-res prediction, will be upsampled for visualization)
            decode_and_save(x_0_low, "x0_low", is_low_res=True)
            
            # Save x0_high_upsampled (upsampled from x0_low)
            decode_and_save(x_0_high_upsampled, "x0_high_upsampled")
            
            # Save x0_high_original (from DiT's predictions, includes NN-filled LR regions)
            decode_and_save(x_0_high_original, "x0_high_original")
            
            # Save x0_merged (final merged prediction)
            decode_and_save(x_0_merged, "x0_merged")
            
            # Also save the foveation mask for reference
            mask_img = (foveation_mask_upsampled.float().cpu().numpy() * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_img)
            mask_pil.save(save_dir / f"step_{step_idx:03d}_foveation_mask.png")
            
            # =====================================================================
            # 6. Save velocity predictions as color-coded maps
            # =====================================================================
            def velocity_to_colormap(velocity_tensor, name, is_low_res=False):
                """
                Convert velocity tensor to color-coded visualization.
                Uses a diverging colormap (blue-white-red) to show negative/zero/positive values.
                
                Args:
                    velocity_tensor: [B, C, H, W] velocity tensor
                    name: Name for saving
                    is_low_res: If True, upsample for visualization
                """
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm
                
                for b_idx in range(velocity_tensor.shape[0]):
                    # Take mean across channels to get a single map
                    v_mean = velocity_tensor[b_idx].mean(dim=0).float().cpu().numpy()  # [H, W]
                    
                    # Also save magnitude (L2 norm across channels)
                    v_magnitude = velocity_tensor[b_idx].float().pow(2).sum(dim=0).sqrt().cpu().numpy()  # [H, W]
                    
                    # Normalize for visualization
                    # For mean: center at 0, use symmetric range
                    v_max = max(abs(v_mean.min()), abs(v_mean.max()), 1e-6)
                    v_mean_normalized = (v_mean / v_max + 1) / 2  # Map to [0, 1]
                    
                    # For magnitude: normalize to [0, 1]
                    mag_max = max(v_magnitude.max(), 1e-6)
                    v_mag_normalized = v_magnitude / mag_max
                    
                    # Apply colormaps
                    # Mean: use RdBu (red-white-blue diverging colormap)
                    colormap_diverging = cm.get_cmap('RdBu_r')  # Red for positive, Blue for negative
                    v_mean_colored = (colormap_diverging(v_mean_normalized)[:, :, :3] * 255).astype(np.uint8)
                    
                    # Magnitude: use viridis or hot
                    colormap_magnitude = cm.get_cmap('hot')
                    v_mag_colored = (colormap_magnitude(v_mag_normalized)[:, :, :3] * 255).astype(np.uint8)
                    
                    # Convert to PIL
                    mean_img = Image.fromarray(v_mean_colored)
                    mag_img = Image.fromarray(v_mag_colored)
                    
                    if is_low_res:
                        mean_img = mean_img.resize((width, height), resample=Image.NEAREST)
                        mag_img = mag_img.resize((width, height), resample=Image.NEAREST)
                    
                    # Save
                    mean_img.save(save_dir / f"step_{step_idx:03d}_{name}_mean_batch_{b_idx}.png")
                    mag_img.save(save_dir / f"step_{step_idx:03d}_{name}_magnitude_batch_{b_idx}.png")
                    
                    # Also save first 3 channels as RGB for direct visualization
                    v_rgb = velocity_tensor[b_idx, :3].float().cpu()  # [3, H, W]
                    # Normalize each channel independently
                    v_rgb_normalized = torch.zeros_like(v_rgb)
                    for c in range(3):
                        c_max = max(abs(v_rgb[c].min()), abs(v_rgb[c].max()), 1e-6)
                        v_rgb_normalized[c] = (v_rgb[c] / c_max + 1) / 2  # Map to [0, 1]
                    v_rgb_img = (v_rgb_normalized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    rgb_img = Image.fromarray(v_rgb_img)
                    if is_low_res:
                        rgb_img = rgb_img.resize((width, height), resample=Image.NEAREST)
                    rgb_img.save(save_dir / f"step_{step_idx:03d}_{name}_rgb_batch_{b_idx}.png")
            
            # Compute v_high_from_upsampler: v = (z_t - x_0) / sigma
            sigma_clamped = sigma.clamp(min=1e-6)
            v_high_from_upsampler = (z_t_high_unpacked - x_0_high_upsampled) / sigma_clamped
            
            # Compute merged velocity
            v_merged = torch.where(
                foveation_mask_upsampled.to(device),
                v_high_original,
                v_high_from_upsampler
            )
            
            # Save velocity maps
            velocity_to_colormap(v_low_unpacked, "v_low", is_low_res=True)
            velocity_to_colormap(v_high_original, "v_high_original")
            velocity_to_colormap(v_high_from_upsampler, "v_high_from_upsampler")
            velocity_to_colormap(v_merged, "v_merged")
            
            # =====================================================================
            # 7. Save x0 latents as color-coded maps (same visualization as velocity)
            # =====================================================================
            velocity_to_colormap(x_0_low, "x0_low_latent", is_low_res=True)
            velocity_to_colormap(x_0_high_upsampled, "x0_high_upsampled_latent")
            velocity_to_colormap(x_0_high_original, "x0_high_original_latent")
            velocity_to_colormap(x_0_merged, "x0_merged_latent")
            
            # =====================================================================
            # 8. Save spatial gradient maps (shows local noisiness/detail)
            # =====================================================================
            def compute_spatial_gradient(tensor, name, is_low_res=False):
                """
                Compute and visualize local spatial gradients to show noisiness.
                
                Uses Sobel-like filters to compute gradients in x and y directions,
                then computes the gradient magnitude. High values indicate high local
                variation (noise or fine detail), low values indicate smooth regions.
                
                Args:
                    tensor: [B, C, H, W] input tensor
                    name: Name for saving
                    is_low_res: If True, upsample for visualization
                """
                import matplotlib.cm as cm
                
                for b_idx in range(tensor.shape[0]):
                    # Get the tensor for this batch item [C, H, W]
                    t = tensor[b_idx].float()
                    
                    # Compute gradients using finite differences (Sobel-like)
                    # Gradient in x direction (horizontal)
                    gx = torch.zeros_like(t)
                    gx[:, :, 1:] = t[:, :, 1:] - t[:, :, :-1]  # [C, H, W]
                    
                    # Gradient in y direction (vertical)
                    gy = torch.zeros_like(t)
                    gy[:, 1:, :] = t[:, 1:, :] - t[:, :-1, :]  # [C, H, W]
                    
                    # Gradient magnitude per channel
                    grad_mag = torch.sqrt(gx.pow(2) + gy.pow(2))  # [C, H, W]
                    
                    # Mean gradient magnitude across channels
                    grad_mag_mean = grad_mag.mean(dim=0).cpu().numpy()  # [H, W]
                    
                    # L2 norm of gradient magnitude across channels
                    grad_mag_l2 = grad_mag.pow(2).sum(dim=0).sqrt().cpu().numpy()  # [H, W]
                    
                    # Normalize for visualization
                    grad_mean_max = max(grad_mag_mean.max(), 1e-6)
                    grad_mean_normalized = grad_mag_mean / grad_mean_max
                    
                    grad_l2_max = max(grad_mag_l2.max(), 1e-6)
                    grad_l2_normalized = grad_mag_l2 / grad_l2_max
                    
                    # Apply colormap (hot: black=smooth, yellow/white=noisy)
                    colormap = cm.get_cmap('hot')
                    grad_mean_colored = (colormap(grad_mean_normalized)[:, :, :3] * 255).astype(np.uint8)
                    grad_l2_colored = (colormap(grad_l2_normalized)[:, :, :3] * 255).astype(np.uint8)
                    
                    # Also create a "noisiness" map using inferno colormap
                    colormap_inferno = cm.get_cmap('inferno')
                    grad_inferno = (colormap_inferno(grad_l2_normalized)[:, :, :3] * 255).astype(np.uint8)
                    
                    # Convert to PIL
                    grad_mean_img = Image.fromarray(grad_mean_colored)
                    grad_l2_img = Image.fromarray(grad_l2_colored)
                    grad_inferno_img = Image.fromarray(grad_inferno)
                    
                    if is_low_res:
                        grad_mean_img = grad_mean_img.resize((width, height), resample=Image.NEAREST)
                        grad_l2_img = grad_l2_img.resize((width, height), resample=Image.NEAREST)
                        grad_inferno_img = grad_inferno_img.resize((width, height), resample=Image.NEAREST)
                    
                    # Save
                    grad_mean_img.save(save_dir / f"step_{step_idx:03d}_{name}_grad_mean_batch_{b_idx}.png")
                    grad_l2_img.save(save_dir / f"step_{step_idx:03d}_{name}_grad_l2_batch_{b_idx}.png")
                    grad_inferno_img.save(save_dir / f"step_{step_idx:03d}_{name}_noisiness_batch_{b_idx}.png")
                    
                    # Also save the raw gradient statistics
                    stats = {
                        'mean_grad': float(grad_mag_mean.mean()),
                        'max_grad': float(grad_mag_mean.max()),
                        'std_grad': float(grad_mag_mean.std()),
                    }
                    
                    # Save gradient direction visualization (shows edge orientation)
                    # Angle of gradient
                    angle = torch.atan2(gy.mean(dim=0), gx.mean(dim=0))  # [H, W]
                    angle_normalized = ((angle + np.pi) / (2 * np.pi)).cpu().numpy()  # Map to [0, 1]
                    
                    colormap_hsv = cm.get_cmap('hsv')
                    angle_colored = (colormap_hsv(angle_normalized)[:, :, :3] * 255).astype(np.uint8)
                    angle_img = Image.fromarray(angle_colored)
                    if is_low_res:
                        angle_img = angle_img.resize((width, height), resample=Image.NEAREST)
                    angle_img.save(save_dir / f"step_{step_idx:03d}_{name}_grad_direction_batch_{b_idx}.png")
            
            # Save spatial gradients for x0 predictions
            compute_spatial_gradient(x_0_low, "x0_low", is_low_res=True)
            compute_spatial_gradient(x_0_high_upsampled, "x0_high_upsampled")
            compute_spatial_gradient(x_0_high_original, "x0_high_original")
            compute_spatial_gradient(x_0_merged, "x0_merged")
            
            # Save spatial gradients for velocity predictions
            compute_spatial_gradient(v_low_unpacked, "v_low", is_low_res=True)
            compute_spatial_gradient(v_high_original, "v_high_original")
            compute_spatial_gradient(v_high_from_upsampler, "v_high_from_upsampler")
            compute_spatial_gradient(v_merged, "v_merged")
            
            # Save spatial gradients for noisy latents z_t (useful for comparison)
            compute_spatial_gradient(z_t_high_unpacked, "z_t_high")
            compute_spatial_gradient(z_t_low_unpacked, "z_t_low", is_low_res=True)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        foveation_mask: Optional[torch.FloatTensor] = None,
        foveation_start_step: int = 0,
        token_upsampler: Optional[torch.nn.Module] = None,
        token_downsampler: Optional[torch.nn.Module] = None,
        save_intermediate_x0: bool = False,
        intermediate_x0_save_dir: Optional[str] = None,
        intermediate_x0_save_steps: Optional[List[int]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        This variant uses X0 (clean image) prediction for upsampling, which is easier
        to learn than upsampling noise directly.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and
                `negative_prompt` is provided.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with `prompt` at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
            foveation_mask (`torch.FloatTensor`, *optional*):
                Binary mask indicating high-resolution regions. Shape: [latent_height, latent_width].
            foveation_start_step (`int`, *optional*, defaults to 0):
                Step at which to start applying foveation.
            token_upsampler (`torch.nn.Module`, *optional*):
                Learned upsampler network for clean image (x0) upsampling.
            token_downsampler (`torch.nn.Module`, *optional*):
                Learned downsampler network for latent downsampling.
            save_intermediate_x0 (`bool`, *optional*, defaults to False):
                If True, saves intermediate x0 (clean image) predictions at each denoising step.
                Requires `intermediate_x0_save_dir` to be set.
            intermediate_x0_save_dir (`str`, *optional*):
                Directory path to save intermediate x0 images. Required if `save_intermediate_x0` is True.
            intermediate_x0_save_steps (`List[int]`, *optional*):
                List of step indices at which to save intermediate x0. If None, saves at all steps.
                Example: [0, 5, 10, 20] to save only at those steps.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

    
        # 4. Prepare high-resolution latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids, resolution_mask, resolution_mask_top_left = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            foveation_mask=None
        )

        # for transformers, we need to pass in the resolution mask and resolution mask top left for RoPE
        self._joint_attention_kwargs["resolution_mask"] = resolution_mask
        self._joint_attention_kwargs["resolution_mask_top_left"] = resolution_mask_top_left

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters


        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        # Helper function to get sigmas from scheduler (matching training code)
        def get_sigmas(timesteps_input, n_dim=4, dtype=torch.float32):
            """
            Extract sigma values from the scheduler for given timesteps.
            This matches the training code's sigma extraction logic.
            """
            scheduler_sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
            schedule_timesteps = self.scheduler.timesteps.to(device)
            timesteps_input = timesteps_input.to(device)
            
            # Find indices of timesteps in the schedule
            step_indices = []
            for t_val in timesteps_input:
                # Find the index where schedule_timesteps equals t_val
                idx = (schedule_timesteps == t_val).nonzero()
                if idx.numel() > 0:
                    step_indices.append(idx.item())
                else:
                    # Fallback: find closest timestep
                    idx = (schedule_timesteps - t_val).abs().argmin().item()
                    step_indices.append(idx)
            
            sigma = scheduler_sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latents_high_res = latents.clone()  # original latents for denoising
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # time embedding
                temb = self.transformer.time_text_embed(timestep.to(dtype=pooled_prompt_embeds.dtype), guidance.to(dtype=pooled_prompt_embeds.dtype), pooled_prompt_embeds)

                # =====================================================================
                # X0 PREDICTION: Extract sigma from scheduler for this timestep
                # =====================================================================
                # In rectified flow: z_t = (1 - sigma) * x_0 + sigma * noise
                # sigma is extracted from the scheduler's sigma schedule
                sigma = get_sigmas(t.unsqueeze(0), n_dim=4, dtype=latents.dtype)
                
                # Get high-res z_t in unpacked format for x0 conversion
                z_t_high_unpacked = self._unpack_latents(
                    latents_high_res,
                    height=self.latent_height * 2 * self.vae_scale_factor,
                    width=self.latent_width * 2 * self.vae_scale_factor,
                    vae_scale_factor=self.vae_scale_factor
                )  # [B, 16, H, W]
                # =====================================================================

                if foveation_mask is not None:
                    # Downsample latents to mixed resolution for DiT input
                    latents = self._downsample_latents(latents, self.latent_height, self.latent_width, foveation_mask=foveation_mask, token_downsampler=token_downsampler, temb=temb)
                    
                    # 2. Further downsample to uniform low-res (averaging high-res blocks)
                    z_t_low_packed = self._downsample_remaining_high_res(
                        latents,
                        self.latent_height,
                        self.latent_width,
                        foveation_mask=foveation_mask
                    )  # [B, H/2 * W/2, 64]
                    # 3. Unpack to spatial format
                    z_t_low_unpacked = self._unpack_latents(
                        z_t_low_packed,
                        height=self.latent_height * self.vae_scale_factor,  # H/2 in pixel space
                        width=self.latent_width * self.vae_scale_factor,    # W/2 in pixel space
                        vae_scale_factor=self.vae_scale_factor
                    )  # [B, 16, H/2, W/2]
                    # =====================================================================
                    latent_image_ids, resolution_mask, resolution_mask_top_left = self._prepare_latent_image_ids(
                        batch_size * num_images_per_prompt, self.latent_height, self.latent_width, device, latents.dtype, foveation_mask
                    )
                    self._joint_attention_kwargs.update({"resolution_mask": resolution_mask, "resolution_mask_top_left": resolution_mask_top_left})
                    assert latents.shape[1] == resolution_mask.shape[0] and latents.shape[1] == resolution_mask_top_left.shape[0]
                else:
                    # No foveation - z_t_low is not needed (function handles this case)
                    z_t_low_unpacked = None

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    
                    # =====================================================================
                    # SAVE INTERMEDIATE X0: Save clean image predictions at this step
                    # =====================================================================
                    if save_intermediate_x0 and foveation_mask is not None:
                        should_save = (intermediate_x0_save_steps is None) or (i in intermediate_x0_save_steps)
                        if should_save and intermediate_x0_save_dir is not None:
                            self.save_intermediate_x0(
                                noise_pred=noise_pred,
                                z_t_high_unpacked=z_t_high_unpacked,
                                z_t_low_unpacked=z_t_low_unpacked,
                                sigma=sigma,
                                foveation_mask=foveation_mask,
                                token_upsampler=token_upsampler,
                                temb=temb,
                                step_idx=i,
                                save_dir=intermediate_x0_save_dir,
                                height=height,
                                width=width,
                            )
                    # =====================================================================
                    
                    # =====================================================================
                    # X0 PREDICTION: Upsample using clean image domain
                    # =====================================================================
                    noise_pred = FluxPipeline_Foveation_X0Pred._upsample_x0_pred_unpacked(
                        noise_pred, 
                        self.latent_height, 
                        self.latent_width, 
                        vae_scale_factor=self.vae_scale_factor,
                        foveation_mask=foveation_mask, 
                        token_upsampler=token_upsampler, 
                        temb=temb,
                        sigma=sigma,
                        z_t_high=z_t_high_unpacked,
                        z_t_low=z_t_low_unpacked,
                    )  # Returns [B, 16, H, W] directly
                    noise_pred = FluxPipeline_Foveation_X0Pred._pack_latents(noise_pred, noise_pred.shape[0], noise_pred.shape[1], noise_pred.shape[2], noise_pred.shape[3])
                    # =====================================================================
                    
                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds

                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        
                        # =====================================================================
                        # X0 PREDICTION: Upsample negative prediction using clean image domain
                        # =====================================================================
                        neg_noise_pred = FluxPipeline_Foveation_X0Pred._upsample_x0_pred_unpacked(
                            neg_noise_pred, 
                            self.latent_height, 
                            self.latent_width, 
                            vae_scale_factor=self.vae_scale_factor,
                            foveation_mask=foveation_mask, 
                            token_upsampler=token_upsampler, 
                            temb=temb,
                            sigma=sigma,
                            z_t_high=z_t_high_unpacked,
                            z_t_low=z_t_low_unpacked,
                        )
                        neg_noise_pred = FluxPipeline_Foveation_X0Pred._pack_latents(neg_noise_pred, neg_noise_pred.shape[0], neg_noise_pred.shape[1], neg_noise_pred.shape[2], neg_noise_pred.shape[3])
                        # =====================================================================
                        
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents_high_res, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            mode = "direct"
            if mode == "direct":
                latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            elif mode == "merge":
                latents = self._downsample_latents(latents, self.latent_height, self.latent_width, foveation_mask=foveation_mask, token_downsampler=token_downsampler, temb=temb)
                # low-res latents
                low_res_latents = self._downsample_remaining_high_res(latents, self.latent_height, self.latent_width, foveation_mask=foveation_mask)
                low_res_latents = self._unpack_latents(low_res_latents, height // 2, width // 2, self.vae_scale_factor)
                low_res_latents = (low_res_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                low_res_image = self.vae.decode(low_res_latents, return_dict=False)[0]
                # high-res latents
                high_res_latents = self._upsample_latents(latents, self.latent_height, self.latent_width, foveation_mask=foveation_mask)
                high_res_latents = self._unpack_latents(high_res_latents, height, width, self.vae_scale_factor)
                high_res_latents = (high_res_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                high_res_image = self.vae.decode(high_res_latents, return_dict=False)[0]
                # upsample low-res image to high-res size
                low_res_image = F.interpolate(low_res_image, size=(height, width), mode='bilinear')
                high_res_foveation_mask = F.interpolate(foveation_mask[None, None, :, :], size=(height, width), mode='bilinear').to(latents.dtype).to(latents.device) # [1, 1, H, W]
                # merge 
                image = low_res_image * (1 - high_res_foveation_mask) + high_res_image * high_res_foveation_mask
            else:
                raise ValueError(f"Output mode {mode} is not supported!")
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
