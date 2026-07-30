# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.utils import logging

from .pipeline_wan_base import WanPipeline as WanPipelineBase
from .utils.helpers import feather_mask_from_bool, transform_base_mask, unified_scheduler_step
from .utils.raft_func import fill_mask_with_raft
from .utils.seg_func import _refine_saliency_latent_mask, refine_mask_w_seg
from .utils.seg_video_func import propagate_mask_w_sam2_tiny


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanPipeline(WanPipelineBase):
    """Wan pipeline with mixed-resolution denoising extensions."""

    def _get_vae_latent_stats(self, device: torch.device, dtype: torch.dtype):
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            device, dtype
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(device, dtype)
        return latents_mean, latents_std

    def _decode_latents_to_video(self, latents: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return latents

        latents = latents.to(self.vae.dtype)
        latents_mean, latents_std = self._get_vae_latent_stats(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        return self.video_processor.postprocess_video(video, output_type=output_type)

    def _resize_x0_to_x0(
        self,
        latents_x0: torch.Tensor,
        resize_scale: float,
        mode: str = "",
        dec_video: bool = False,
    ):
        """Resize latent x0 through the learned latent-space resizers."""

        if resize_scale == 2.0:
            assert mode in ["", "repeat", "bi"]
            if mode == "": 
                new_lt = self.lt_upscaler(latents_x0)
            elif mode == "bi": 
                new_lt = self.bi_upscaler(latents_x0)
            elif mode == "repeat": 
                new_lt = latents_x0.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        elif resize_scale == 0.5:
            assert mode in [""]
            if mode == "":
                new_lt = self.lt_downscaler(latents_x0)

        video = None
        if dec_video:
            video = self.tvae.decode_video(latents_x0.transpose(1, 2), parallel=False, show_progress_bar=False).transpose(1, 2).float().mul_(2).sub_(1)
        
        return new_lt, video

    def _add_noise_at_t(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = timestep.reshape(1).to(x0.device, dtype=torch.float32)[0] * 1e-3
        return (sigma * noise.to(torch.float32) + (1.0 - sigma) * x0.to(torch.float32)).to(x0.dtype)

    def _synthesize_latents_from_x0(
        self,
        latents_fake_x0: torch.Tensor,
        latents_ref: Optional[torch.Tensor],
        latents_noise: torch.Tensor,
        mask_bool: Optional[torch.Tensor],
        target_timestep: torch.Tensor,
    ) -> torch.Tensor:
        latents_fake = self._add_noise_at_t(latents_fake_x0, latents_noise, target_timestep)

        if mask_bool is not None and latents_ref is not None:
            mask_latent = mask_bool.to(device=latents_fake.device, dtype=latents_fake.dtype)
            latents_fake = latents_fake * mask_latent + latents_ref * (1.0 - mask_latent)

        return latents_fake

    def _get_saliency(
        self,
        video_lr_down_uint8: torch.Tensor,
        lt_lr_shape: torch.Tensor,
    ):
        video_lr_down_uint8_bt = rearrange(video_lr_down_uint8, "b c t h w -> (b t) c h w")

        centerbias_tensor = self.centerbias_tensor.clone()
        centerbias_tensor = centerbias_tensor.expand(video_lr_down_uint8_bt.shape[0], -1, -1)

        log_density_prediction = self.gaze_model(video_lr_down_uint8_bt, centerbias_tensor)
        density_preds = torch.exp(log_density_prediction)
        if density_preds.dim() == 3:
            density_preds = density_preds.unsqueeze(1)
        density_preds = rearrange(
            density_preds, "(b t) c h w -> b c t h w", b=video_lr_down_uint8.shape[0], t=video_lr_down_uint8.shape[2],
        )
        density_preds = density_preds.squeeze(1)
        saliency_map = F.interpolate(
            density_preds,
            size=(lt_lr_shape[-2] // 2, lt_lr_shape[-1] // 2),
        )
        return saliency_map

    def _refine_mask_tokens_with_seg(
        self,
        mask_tokens: torch.Tensor,
        video_lr_down_uint8: torch.Tensor,
        hr_ratio: float,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size, t_tokens, latent_h, latent_w = mask_tokens.shape
        _, _, _, video_h, video_w = video_lr_down_uint8.shape
        patch_h, patch_w = video_h // latent_h, video_w // latent_w
        target_k = latent_h * latent_w * hr_ratio * 1.3
        base_idx = t_tokens // 2

        if batch_size != 1:
            raise ValueError("Segmentation-guided mask refinement currently expects batch size 1.")

        base_refined_mask = refine_mask_w_seg(
            model=self.seg_model,
            img_tensor=video_lr_down_uint8[:, :, base_idx],
            mask_tensor=mask_tokens[:, base_idx : base_idx + 1],
            target_k=target_k,
            conf=0.25,
            overlap_threshold=0.1,
        )

        if t_tokens == 1:
            return base_refined_mask

        base_refined_mask_hw = base_refined_mask[:, 0].repeat_interleave(patch_h, dim=1).repeat_interleave(
            patch_w, dim=2
        )
        refined_video_mask = propagate_mask_w_sam2_tiny(
            predictor=self.sam2_predictor,
            video_tensor_bcthw=video_lr_down_uint8,
            base_mask_bhw=base_refined_mask_hw,
            base_idx=base_idx,
            device=device,
        )
        refined_latent_mask = refined_video_mask.reshape(
            batch_size, t_tokens, latent_h, patch_h, latent_w, patch_w
        ).sum(dim=(3, 5)) > (0.05 * patch_h * patch_w)

        refined_latent_mask_np = refined_latent_mask.detach().cpu().numpy()
        for batch_idx in range(refined_latent_mask_np.shape[0]):
            for frame_idx in range(refined_latent_mask_np.shape[1]):
                refined_latent_mask_np[batch_idx, frame_idx] = _refine_saliency_latent_mask(
                    refined_latent_mask_np[batch_idx, frame_idx],
                    target_k=target_k,
                )

        refined_latent_mask = torch.from_numpy(refined_latent_mask_np).to(device=device)
        _r = refined_latent_mask.float().mean()
        if _r > 0.1 and _r < 0.9:
            print("use seg mask")
            return refined_latent_mask
        return mask_tokens

    def _initialize_mask_tokens(
        self,
        saliency_map: torch.Tensor,
        video_lr_down: torch.Tensor,
        video_lr_down_uint8: torch.Tensor,
        hr_ratio: float,
        use_seg: bool,
        use_raft: bool,
        device: torch.device,
    ) -> torch.Tensor:
        importance_flat = saliency_map.reshape(saliency_map.shape[0], -1)
        _, importance_idx = torch.topk(
            importance_flat,
            int(hr_ratio * importance_flat.shape[1]),
            dim=1,
            largest=True,
        )
        mask_tokens = torch.zeros_like(importance_flat, dtype=torch.bool)
        mask_tokens.scatter_(1, importance_idx, True)
        mask_tokens = mask_tokens.reshape_as(saliency_map)

        if use_seg:
            # 253 ms on H100
            return self._refine_mask_tokens_with_seg(mask_tokens, video_lr_down_uint8, hr_ratio, device)
        elif use_raft:
            print("use raft")
            return fill_mask_with_raft(video_lr_down, mask_tokens, self.raft)
        return mask_tokens

    def _prepare_boundary_masks(
        self,
        mask_tokens: torch.Tensor,
        hr_scale: int,
        pad1: int,
        pad2: int,
        latents_lr_dtype: torch.dtype,
        latents_hr_dtype: torch.dtype,
    ):
        core_mask = mask_tokens.clone()
        mask_lr_keep_tokens = ~mask_tokens
        bd1_lr_mask = None
        bd2_hr_mask = None

        if pad1 > 0 and mask_lr_keep_tokens.numel() > 0:
            mask_lr_keep_tokens = (
                F.max_pool3d(
                    mask_lr_keep_tokens.float().unsqueeze(1),
                    kernel_size=(3, pad1 * 2 + 1, pad1 * 2 + 1),
                    stride=1,
                    padding=(1, pad1, pad1),
                )
                .squeeze(1)
                .to(torch.bool)
            )
            boundary1_mask_tokens = mask_lr_keep_tokens & mask_tokens
            bd1_lr_mask = transform_base_mask(boundary1_mask_tokens, 1, self.transformer.config.patch_size).to(latents_lr_dtype)

        if pad2 > 0:
            dilated_mask_tokens = (
                F.max_pool3d(
                    mask_tokens.float().unsqueeze(1),
                    kernel_size=(3, pad2 * 2 + 1, pad2 * 2 + 1),
                    stride=1,
                    padding=(1, pad2, pad2),
                )
                .squeeze(1)
                .to(torch.bool)
            )
            boundary2_mask_tokens = dilated_mask_tokens & (~mask_tokens)
            mask_tokens = dilated_mask_tokens
            bd2_hr_mask = transform_base_mask(boundary2_mask_tokens, hr_scale, self.transformer.config.patch_size).to(latents_hr_dtype)

        n_lr = mask_lr_keep_tokens.sum(dim=(1, 2, 3))
        n_hr = mask_tokens.sum(dim=(1, 2, 3)) * (hr_scale * hr_scale)
        if not torch.equal(n_lr, n_lr[0].expand_as(n_lr)) or not torch.equal(n_hr, n_hr[0].expand_as(n_hr)):
            raise ValueError("All batch elements must have equal numbers of selected LR/HR tokens.")

        print(f"mask ratio: {core_mask.float().mean():.2f}")

        return {
            "core_mask": core_mask,
            "mask_lr_keep_tokens": mask_lr_keep_tokens,
            "mask_tokens": mask_tokens,
            "n_hr_tokens_int": int(n_hr[0].item()),
            "n_lr_tokens_int": int(n_lr[0].item()),
            "bd1_lr_mask": bd1_lr_mask,
            "bd2_hr_mask": bd2_hr_mask,
        }

    def _decode_mixed_res_video(
        self,
        latents_lr: torch.Tensor,
        latents_hr: torch.Tensor,
        core_mask: torch.Tensor,
        mask_tokens: torch.Tensor,
        hr_scale: int,
        output_type: str,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ):
        latents_lr = latents_lr.to(self.vae.dtype)
        latents_hr = latents_hr.to(self.vae.dtype)

        dec_lr = self.vae.decode(latents_lr / latents_std + latents_mean, return_dict=False)[0]
        _, _, num_frames_px, _, _ = dec_lr.shape

        _, _, _, h_latent_hr, w_latent_hr = latents_hr.shape
        hr_px_height = h_latent_hr * self.vae_scale_factor_spatial
        hr_px_width = w_latent_hr * self.vae_scale_factor_spatial
        dec_lr_up = F.interpolate(
            dec_lr, size=(num_frames_px, hr_px_height, hr_px_width), mode="trilinear", align_corners=False
        ).clamp_(-1.0, 1.0)

        posterior_fake_hr = self.vae.encode(dec_lr_up)
        latents_fake_hr_raw = posterior_fake_hr.latent_dist.sample()
        latents_fake_hr = (latents_fake_hr_raw - latents_mean) * latents_std

        mask_hr_latent = transform_base_mask(core_mask, hr_scale, self.transformer.config.patch_size).to(
            latents_fake_hr.dtype
        )
        latents_blended = latents_fake_hr * (1.0 - mask_hr_latent) + latents_hr * mask_hr_latent

        latents_dec_input = latents_blended / latents_std + latents_mean
        video = self.vae.decode(latents_dec_input, return_dict=False)[0]
        return self.video_processor.postprocess_video(video, output_type=output_type)

    def _clear_mixed_mask_cache(self):
        if self.transformer is not None:
            self.transformer._mixed_mask_cache = None
        if self.transformer_2 is not None:
            self.transformer_2._mixed_mask_cache = None

    @torch.no_grad()
    def warmup_seg(self, n_frames, lr_height, lr_width, device, autocast_dtype):
        for _ in range(2):
            self.seg_model.predict([np.zeros((lr_height, lr_width, 3), dtype=np.uint8)], verbose=False)
        if n_frames > 1:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast_dtype):
                state = self.sam2_predictor.init_state(
                    torch.zeros((3, n_frames // 4, lr_height, lr_width), dtype=torch.uint8, device=device)
                )
                self.sam2_predictor.reset_state(state)
                self.sam2_predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=1, 
                    mask=torch.ones((lr_height, lr_width), dtype=torch.bool, device=device),
                )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # Mixed-resolution options
        mixed_res: bool = True,
        hr_ratio: float = 0.25,
        hr_scale: int = 2,
        stg1_true_steps: int = 0,
        pad1: int = 2,
        pad2: int = 1,
        mask_type: str = "w_raft",
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor`.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                Callback support is delegated to the base pipeline when `mixed_res=False`. The mixed-resolution path
                currently requires this to be `None`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                Tensor inputs for `callback_on_step_end`.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder.
            mixed_res (`bool`, defaults to `False`):
                Enable the mixed-resolution denoising path that combines low- and high-resolution latent tokens.
            hr_ratio (`float`, defaults to `0.25`):
                Probability of selecting a latent token for high-resolution refinement when `mixed_res` is enabled.
            hr_scale (`int`, defaults to `2`):
                Spatial upscaling factor between low-resolution and high-resolution latent grids.
            stg1_true_steps (`int`, defaults to `0`):
                Number of initial low-resolution denoising steps before constructing the mixed-resolution branch. The
                mixed-resolution path requires at least one LR step to estimate `x0`, so values below `1` are treated
                as `1`.

        Returns:
            [`~WanPipelineOutput`] or `tuple`
        """

        if not mixed_res:
            return super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

        if callback_on_step_end is not None:
            raise ValueError("`callback_on_step_end` is not supported when `mixed_res=True`.")

        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        use_seg, use_raft = False, False # mask_type == "base"
        if mask_type == "w_raft":
            use_raft = True
        elif mask_type == "w_seg":
            use_seg = True

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = (
            self.transformer.config.in_channels if self.transformer is not None else self.transformer_2.config.in_channels
        )
        warmup_steps = max(int(stg1_true_steps or 0), 1)
        mixed_res_active = False

        bsz = batch_size * num_videos_per_prompt
        latents_lr_noise = self.prepare_latents(
            bsz,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        latents_hr_noise = self.prepare_latents(
            bsz,
            num_channels_latents,
            height * hr_scale,
            width * hr_scale,
            num_frames,
            torch.float32,
            device,
            generator,
            None,
        )
        ones_lr = torch.ones(latents_lr_noise.shape, dtype=torch.float32, device=device)
        latents_mean, latents_std = self._get_vae_latent_stats(device, self.vae.dtype)

        latents_lr = latents_lr_noise.clone()
        latents_hr = None
        latents_lr_x0 = None
        core_mask = None
        mask_lr_keep_tokens = None
        mask_tokens = None
        n_lr_tokens_int = None
        n_hr_tokens_int = None
        bd1_lr_mask = None
        bd2_hr_mask = None

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        current_model = self.transformer
        current_guidance_scale = guidance_scale

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if not mixed_res_active:
                    latent_model_input = latents_lr.to(transformer_dtype)
                    if self.config.expand_timesteps:
                        temp_ts = (ones_lr[0][0][:, ::2, ::2] * t).flatten()
                        timestep = temp_ts.unsqueeze(0).expand(bsz, -1)
                    else:
                        timestep = t.expand(bsz)

                    with current_model.cache_context("cond"):
                        noise_pred = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]

                    if self.do_classifier_free_guidance:
                        with current_model.cache_context("uncond"):
                            noise_uncond = current_model(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                    prev_latents_and_x0 = self.scheduler.step(noise_pred, t, latents_lr, return_dict=False)
                    if len(prev_latents_and_x0) == 2:
                        latents_lr, latents_lr_x0 = prev_latents_and_x0
                    else:
                        latents_lr, latents_lr_x0 = prev_latents_and_x0[0], None

                    should_activate = warmup_steps > 0 and (i + 1) == warmup_steps and (i + 1) < len(timesteps)
                    if should_activate:
                        if latents_lr_x0 is None:
                            raise ValueError("Mixed-resolution warmup requires the scheduler to return an `x0` estimate.")

                        next_t = timesteps[i + 1].expand(bsz)
                        latents_fake_hr_x0, video_lr = self._resize_x0_to_x0(latents_lr_x0, hr_scale, mode="", dec_video=True)
                        latents_hr = self._add_noise_at_t(latents_fake_hr_x0, latents_hr_noise, next_t)
                        video_lr_down = torch.cat([video_lr[:, :, :1], video_lr[:, :, 2::4]], dim=2) if video_lr.shape[2] > 1 else video_lr
                        video_lr_down_uint8 = ((video_lr_down * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
                        saliency_map = self._get_saliency(video_lr_down_uint8, latents_lr_x0.shape)

                        mask_tokens = self._initialize_mask_tokens(
                            saliency_map=saliency_map,
                            video_lr_down=video_lr_down,
                            video_lr_down_uint8=video_lr_down_uint8,
                            hr_ratio=hr_ratio,
                            use_seg=use_seg,
                            use_raft=use_raft,
                            device=device,
                        )
                        boundary_masks = self._prepare_boundary_masks(
                            mask_tokens=mask_tokens,
                            hr_scale=hr_scale,
                            pad1=pad1,
                            pad2=pad2,
                            latents_lr_dtype=latents_lr.dtype,
                            latents_hr_dtype=latents_hr.dtype,
                        )
                        core_mask = boundary_masks["core_mask"]
                        mask_lr_keep_tokens = boundary_masks["mask_lr_keep_tokens"]
                        mask_tokens = boundary_masks["mask_tokens"]
                        n_hr_tokens_int = boundary_masks["n_hr_tokens_int"]
                        n_lr_tokens_int = boundary_masks["n_lr_tokens_int"]
                        bd1_lr_mask = boundary_masks["bd1_lr_mask"]
                        bd2_hr_mask = boundary_masks["bd2_hr_mask"]
                        mixed_res_active = True

                        if self.transformer_2 is not None:
                            current_model = self.transformer_2
                            current_guidance_scale = guidance_scale_2
                else:
                    hs_lr = latents_lr.to(transformer_dtype)
                    hs_hr = latents_hr.to(transformer_dtype)

                    if self.config.expand_timesteps:
                        if n_lr_tokens_int is None or n_hr_tokens_int is None:
                            raise ValueError("Mixed-resolution timestep expansion requires token counts.")
                        timestep = t.reshape(1, 1).expand(bsz, n_lr_tokens_int + n_hr_tokens_int)
                    else:
                        timestep = t.expand(bsz)

                    with current_model.cache_context("cond"):
                        noise_pred_lr_c, noise_pred_hr_c = current_model.forward_mixed(
                            hidden_states_lr=hs_lr,
                            hidden_states_hr=hs_hr,
                            mask_lr_tokens=mask_tokens,
                            mask_lr_keep_tokens=mask_lr_keep_tokens,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=None,
                            attention_kwargs=attention_kwargs,
                        )

                    if self.do_classifier_free_guidance:
                        with current_model.cache_context("uncond"):
                            noise_pred_lr_u, noise_pred_hr_u = current_model.forward_mixed(
                                hidden_states_lr=hs_lr,
                                hidden_states_hr=hs_hr,
                                mask_lr_tokens=mask_tokens,
                                mask_lr_keep_tokens=mask_lr_keep_tokens,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                encoder_hidden_states_image=None,
                                attention_kwargs=attention_kwargs,
                            )
                        noise_pred_lr = noise_pred_lr_u + current_guidance_scale * (noise_pred_lr_c - noise_pred_lr_u)
                        noise_pred_hr = noise_pred_hr_u + current_guidance_scale * (noise_pred_hr_c - noise_pred_hr_u)
                    else:
                        noise_pred_lr, noise_pred_hr = noise_pred_lr_c, noise_pred_hr_c

                    latents_lr_new, latents_hr_new, latents_lr_x0, latents_hr_x0 = unified_scheduler_step(
                        self.scheduler,
                        t,
                        latents_lr,
                        latents_hr,
                        noise_pred_lr,
                        noise_pred_hr,
                    )

                    if pad2 > 0 and pad1 > 0 and (i + 1) < len(timesteps):
                        next_t = timesteps[i + 1].expand(bsz)

                        latents_fake_hr_x0, _ = self._resize_x0_to_x0(latents_lr_x0, hr_scale, "bi")
                        latents_hr = self._synthesize_latents_from_x0(
                            latents_fake_hr_x0,
                            latents_hr_new,
                            latents_hr_noise,
                            bd2_hr_mask,
                            next_t,
                        )

                        latents_fake_lr_x0, _ = self._resize_x0_to_x0(latents_hr_x0, 1.0 / hr_scale, "")
                        latents_lr = self._synthesize_latents_from_x0(
                            latents_fake_lr_x0,
                            latents_lr_new,
                            latents_lr_noise,
                            bd1_lr_mask,
                            next_t,
                        )
                    else:
                        latents_hr = latents_hr_new
                        latents_lr = latents_lr_new

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None
        self._clear_mixed_mask_cache()

        if latents_hr is None: # LR only
            video = self._decode_latents_to_video(latents_lr, output_type=output_type)
        else:
            if pad1 <= 0 or pad2 <= 0:
                raise ValueError("Mixed-resolution output blending requires both overlap pads to be positive.")
            video = self._decode_mixed_res_video(
                latents_lr=latents_lr,
                latents_hr=latents_hr,
                core_mask=core_mask,
                mask_tokens=mask_tokens,
                hr_scale=hr_scale,
                output_type=output_type,
                latents_mean=latents_mean,
                latents_std=latents_std,
            )

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
