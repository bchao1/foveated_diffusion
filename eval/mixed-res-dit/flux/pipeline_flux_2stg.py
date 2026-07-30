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

from __future__ import annotations
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy

from einops import rearrange
import numpy as np

import torch
import torch.nn.functional as F
from diffusers.utils import logging

from .pipeline_flux_base import (
    calculate_shift, 
    retrieve_timesteps, 
    FluxPipelineOutput,
    PipelineImageInput,
    FluxPipeline as FluxPipeline_BASE,
)
from .utils.dct_func import patch_dct_energy_mask
from .utils.seg_func import refine_mask_w_seg
from .utils.helpers import transform_base_mask, feather_mask_from_bool, quick_benchmark_gpu, align_latent_mean_shift_flexible


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class FluxPipeline(FluxPipeline_BASE):

    def _vae1_decode(self, latents, hw_scale=1, need_unpack=True) -> torch.Tensor:
        if need_unpack:
            latents = self._unpack_latents(latents, self._h*hw_scale, self._w*hw_scale, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]     
        return image   

    def _tvae_decode(self, latents, hw_scale=1, need_unpack=True) -> torch.Tensor:
        if need_unpack:
            latents = self._unpack_latents(latents, self._h*hw_scale, self._w*hw_scale, self.vae_scale_factor)
        image = self.tvae.decode(latents, return_dict=False)[0]     
        return image   

    def _pack_lt_w_scale(self, latents, hw_scale) -> torch.Tensor:
        lt_h = 2 * (int(self._h*hw_scale) // (self.vae_scale_factor * 2))
        lt_w = 2 * (int(self._w*hw_scale) // (self.vae_scale_factor * 2))
        latents = self._pack_latents(latents, latents.shape[0], self._c_lt, lt_h, lt_w)
        return latents # [B, 4096, 64]

    def _resize_x0_to_x0(
        self,
        latents_x0: torch.Tensor,
        resize_scale: float,
        mode: str = "",
        ori_scale: int = 1,
        dec_image: bool = False,
        align_mean: bool = True,
    ) -> torch.Tensor:
        """to obtain fake new-res latents x0"""
        
        assert resize_scale in [0.5, 2.0]
        new_scale = resize_scale * ori_scale

        lt = self._unpack_latents(latents_x0, self._h*ori_scale, self._w*ori_scale, self.vae_scale_factor)
        if resize_scale == 2.0:
            assert mode in ["", "repeat", "bi"]
            if mode == "": 
                new_lt = self.lt_upscaler(lt)
            elif mode == "bi": 
                new_lt = self.bi_upscaler(lt)
            elif mode == "repeat": 
                new_lt = lt.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        elif resize_scale == 0.5:
            assert mode in ["", "repeat"]
            if mode == "":
                new_lt = self.lt_downscaler(lt)
            elif mode == "repeat":
                new_lt = lt[..., ::2, ::2]
        if align_mean:
            new_lt, _ = align_latent_mean_shift_flexible(lt, new_lt)
        latents_new_x0 = self._pack_lt_w_scale(new_lt, new_scale) 

        # quick_benchmark_gpu(self._tvae_decode, latents_x0, ori_scale, iters=50, warmup=20, name='tvae')
        image = self._tvae_decode(latents_x0, ori_scale) if dec_image else None

        return latents_new_x0, image

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
        latents_orig: Optional[torch.Tensor],
        latents_noise: torch.Tensor,
        mask_bool: Optional[torch.Tensor],
        target_timestep: torch.Tensor,
    ) -> torch.Tensor:
        latents_fake = self._add_noise_at_t(latents_fake_x0, latents_noise, target_timestep)

        if mask_bool is not None and latents_orig is not None:
            mask_latent = mask_bool.to(latents_orig.dtype)
            latents_fake = latents_fake * mask_latent + latents_orig * (1.0 - mask_latent)

        return latents_fake

    def _get_saliency(
        self,
        img_lr: torch.Tensor,
    ):
        # print(img_lr.shape, img_lr.min(), img_lr.max(), img_lr.dtype) 
        # torch.Size([1, 3, 512, 512]) tensor(-1.2500, device='cuda:0', dtype=torch.bfloat16) tensor(1.2969, device='cuda:0', dtype=torch.bfloat16) torch.bfloat16
        saliency_map, img_lr_uint8 = None, None
        
        img_lr_uint8 = ((img_lr.float() * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        img_lr_uint8 = F.interpolate(img_lr_uint8, self.gaze_shape)
        centerbias_tensor = self.centerbias_tensor.clone()
        centerbias_tensor = centerbias_tensor.expand(img_lr_uint8.shape[0], -1, -1)

        log_density_prediction = self.gaze_model(img_lr_uint8, centerbias_tensor)
        density_preds = torch.exp(log_density_prediction)
        if density_preds.ndim == 3:
            density_preds = density_preds.unsqueeze(1)
        lt_h = self._h // (self.vae_scale_factor * 2)
        lt_w = self._w // (self.vae_scale_factor * 2)
        saliency_map = F.interpolate(density_preds, size=(lt_h, lt_w))

        return saliency_map, img_lr_uint8

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents_lr: Optional[torch.FloatTensor] = None,
        latents_hr: Optional[torch.FloatTensor] = None,
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
        # Mixed-resolution options
        hr_ratio: float = 0.25,
        hr_scale: int = 2,
        pad1: int = 2,
        pad2: int = 2, 
        mid_up_mode1: str = "repeat",
        mid_up_mode2: str = "",
        stg1_true_steps: int = 6,
        stg2_true_steps: int = 9,
        use_dct_mid_up_split: bool = True,
        mask_type: str = "w_seg",
        feather_radius: int = 28,
    ):
        if num_images_per_prompt != 1:
            raise ValueError("`num_images_per_prompt` must be 1 in this pipeline.")
        assert callback_on_step_end is None

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._h, self._w = height, width

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
        num_inference_steps = stg1_true_steps + stg2_true_steps
        self._num_timesteps = num_inference_steps

        use_seg = False # mask_type == "base"
        if mask_type == "w_seg":
            use_seg = True

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        if do_true_cfg:
            raise ValueError("`true_cfg_scale` path is not implemented in this pipeline.")

        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        self._c_lt = num_channels_latents
        bsz = batch_size * num_images_per_prompt

        latents_lr_noise, latent_lr_image_ids = self.prepare_latents(
            bsz,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents_lr,
        )
        latents_hr_noise, latent_hr_image_ids = self.prepare_latents(
            bsz,
            num_channels_latents,
            height * hr_scale,
            width * hr_scale,
            prompt_embeds.dtype,
            device,
            generator,
            latents_hr,
        )
        stage2_lr_noise = latents_lr_noise

        image_seq_len_stg1 = latents_hr_noise.shape[1]

        p_h, p_w = 2, 2
        ht = (height // self.vae_scale_factor) // p_h
        wt = (width // self.vae_scale_factor) // p_w
        if ht * wt != latents_lr_noise.shape[1]:
            raise ValueError("Invalid LR token shape for mask construction.")

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(bsz)
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

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                bsz,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            _ = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                bsz,
            )

        mu = calculate_shift(
            image_seq_len_stg1,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        scheduler_stg2_lr = self.scheduler
        scheduler_stg2_hr = self.scheduler_hr
        stg2_start_idx_lr = num_inference_steps - stg2_true_steps
        stg2_start_idx_hr = num_inference_steps - stg2_true_steps

        timesteps_lr, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        timesteps_stg2_hr_full, _ = retrieve_timesteps(
            scheduler_stg2_hr,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )

        timesteps_stg1 = timesteps_lr[:stg1_true_steps]
        timesteps_stg2_lr = timesteps_lr[stg2_start_idx_lr:]
        timesteps_stg2_hr = timesteps_stg2_hr_full[stg2_start_idx_hr:]
        if len(timesteps_stg2_lr) != stg2_true_steps or len(timesteps_stg2_hr) != stg2_true_steps:
            raise RuntimeError("Invalid stage-2 timestep slicing.")

        latents_lr = latents_lr_noise.clone()
        latents_hr = None
        latents_lr_x0 = None
        latents_hr_x0 = None
        saliency_map = None

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=stg1_true_steps + stg2_true_steps) as progress_bar:
            # Stage-1: LR denoise only
            for i, t in enumerate(timesteps_stg1):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                timestep = t.expand(bsz).to(latents_lr_noise.dtype)
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents_lr,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_lr_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                latents_lr, latents_lr_x0 = self.scheduler.step(noise_pred, t, latents_lr, return_dict=False)

                progress_bar.update()

            # Stage-1 artifacts: hr_x0 / saliency / masks
            latents_hr_x0, stage1_img_lr = self._resize_x0_to_x0(latents_lr_x0, hr_scale, mode=mid_up_mode1, ori_scale=1, dec_image=True)
            latents_hr_x0 = latents_hr_x0.to(latents_lr.dtype)

            saliency_map, stage1_img_lr_uint8 = self._get_saliency(stage1_img_lr)
            imp_flat = saliency_map.view(bsz, -1)
            target_k = int(hr_ratio * imp_flat.shape[1])
            _, imp_idx = torch.topk(imp_flat, target_k, dim=1, largest=True)
            imp_mask = torch.zeros_like(imp_flat, dtype=torch.bool)
            imp_mask.scatter_(1, imp_idx, True)
            core_hr_mask_LR = imp_mask.view(*saliency_map.shape) # core hr token areas, in LR latent

            if use_seg:
                # 16 ms on H100
                core_hr_mask_LR_w_seg = refine_mask_w_seg(
                    self.seg_model, stage1_img_lr_uint8, core_hr_mask_LR, target_k*1.1, conf=0.05, pool_threshold=0.05)
                
                tmp_mask_lr = (~core_hr_mask_LR_w_seg).float()
                if pad1 > 0:
                    tmp_mask_lr = F.max_pool2d(tmp_mask_lr, kernel_size=pad1 * 2 + 1, stride=1, padding=pad1) 
                tmp_mask_hr = core_hr_mask_LR_w_seg.float()
                if pad2 > 0:
                    tmp_mask_hr = F.max_pool2d(tmp_mask_hr, kernel_size=pad2 * 2 + 1, stride=1, padding=pad2)
                if tmp_mask_lr.mean() < 0.95 and tmp_mask_hr.mean() < 0.95:
                    core_hr_mask_LR = core_hr_mask_LR_w_seg
                    print("use seg mask")
            
            print(f"mask ratio: {core_hr_mask_LR.float().mean():.2f}")

            actual_lr_mask_bchw = ~core_hr_mask_LR
            if pad1 > 0:
                actual_lr_mask_bchw = (
                    F.max_pool2d(
                        actual_lr_mask_bchw.float(),
                        kernel_size=pad1 * 2 + 1,
                        stride=1,
                        padding=pad1,
                    ).to(torch.bool)
                )
                bd1_mask_bchw = core_hr_mask_LR 
                bd1_mask = transform_base_mask(bd1_mask_bchw, 1, patch_size=1).to(latents_lr.dtype)
                bd1_mask = rearrange(bd1_mask, "b c h w -> b (h w) c")

            actual_hr_mask_bchw_LR = copy.deepcopy(core_hr_mask_LR)
            if pad2 > 0:
                actual_hr_mask_bchw_LR = (
                    F.max_pool2d(
                        core_hr_mask_LR.float(),
                        kernel_size=pad2 * 2 + 1,
                        stride=1,
                        padding=pad2,
                    ).to(torch.bool)
                )
                bd2_mask_bchw_LR = ~core_hr_mask_LR # Outside the core region
                bd2_mask = transform_base_mask(bd2_mask_bchw_LR, hr_scale, patch_size=1).to(latents_hr_x0.dtype)
                bd2_mask = rearrange(bd2_mask, "b c h w -> b (h w) c")

            # (1, 1, 32, 32) -> (1, 32, 32)   (1, 64, 64)
            actual_lr_mask = transform_base_mask(actual_lr_mask_bchw, 1, patch_size=1)
            actual_hr_mask = transform_base_mask(actual_hr_mask_bchw_LR, hr_scale, patch_size=1)
            # -> (1, 1024,1)   (1, 4096, 1)
            actual_lr_mask = rearrange(actual_lr_mask.to(latents_lr.dtype), "b c h w -> b (h w) c")
            actual_hr_mask = rearrange(actual_hr_mask.to(latents_lr.dtype), "b c h w -> b (h w) c")

            if use_dct_mid_up_split:
                dct_high_mask = patch_dct_energy_mask(
                    stage1_img_lr.float(),
                    threshold = 0.15,
                )[1]
                assert dct_high_mask.shape == core_hr_mask_LR.shape

                low_energy_core_mask_lr = core_hr_mask_LR & (~dct_high_mask)
                mode2_mask_hr = transform_base_mask(low_energy_core_mask_lr, hr_scale, patch_size=1)

                latents_hr_x0_mode2, _ = self._resize_x0_to_x0(latents_lr_x0, hr_scale, mode=mid_up_mode2, ori_scale=1, dec_image=False)
                latents_hr_x0_mode2 = latents_hr_x0_mode2.to(latents_lr.dtype)
                mode2_mask_hr = rearrange(mode2_mask_hr, "b c h w -> b (h w) c").to(latents_hr_x0.dtype)
                latents_hr_x0 = latents_hr_x0 * (1.0 - mode2_mask_hr) + latents_hr_x0_mode2 * mode2_mask_hr

            # Stage-2 init: re-noise from x0 at timestep index (stg2_tot_steps_* - stg2_true_steps)
            stage2_start_t_lr = timesteps_stg2_lr[0]
            stage2_start_t_hr = timesteps_stg2_hr[0]

            latents_hr = self._add_noise_at_t(latents_hr_x0, latents_hr_noise, stage2_start_t_hr)
            latents_lr = self._add_noise_at_t(latents_lr_x0, stage2_lr_noise, stage2_start_t_lr)

            # Stage-2: mixed-resolution denoise
            scheduler_stg2_lr.set_begin_index(stg2_start_idx_lr)
            scheduler_stg2_hr.set_begin_index(stg2_start_idx_hr)
            for j, (t_lr, t_hr) in enumerate(zip(timesteps_stg2_lr, timesteps_stg2_hr)):
                if self.interrupt:
                    continue

                self._current_timestep = t_lr
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                timestep_lr = t_lr.expand(bsz).to(stage2_lr_noise.dtype)
                timestep_hr = t_hr.expand(bsz).to(stage2_lr_noise.dtype)
                next_t_lr = timesteps_stg2_lr[j + 1] if (j + 1) < len(timesteps_stg2_lr) else None
                next_t_hr = timesteps_stg2_hr[j + 1] if (j + 1) < len(timesteps_stg2_hr) else None

                with self.transformer.cache_context("cond"):
                    noise_pred_lr, noise_pred_hr = self.transformer.forward_mixed(
                        hidden_states_lr=latents_lr,
                        hidden_states_hr=latents_hr,
                        mask_lr_tokens=actual_hr_mask_bchw_LR,
                        mask_lr_keep_tokens=actual_lr_mask_bchw,
                        timestep_lr=timestep_lr / 1000,
                        timestep_hr=timestep_hr / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids_lr=latent_lr_image_ids,
                        img_ids_hr=latent_hr_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                    )

                latents_lr_new, latents_lr_x0_new = scheduler_stg2_lr.step(
                    noise_pred_lr,
                    t_lr,
                    latents_lr,
                    return_dict=False,
                )
                latents_hr_new, latents_hr_x0_new = scheduler_stg2_hr.step(
                    noise_pred_hr,
                    t_hr,
                    latents_hr,
                    return_dict=False,
                )

                if (pad1 > 0) and (pad2 > 0) and next_t_hr is not None:
                    lt_up_lr_x0, _ = self._resize_x0_to_x0(latents_lr_x0_new, hr_scale, "bi", ori_scale=1)
                    latents_hr = self._synthesize_latents_from_x0(
                        lt_up_lr_x0,
                        latents_hr_new,
                        latents_hr_noise,
                        bd2_mask,
                        next_t_hr,
                    )
                else:
                    latents_hr = latents_hr_new
                    latents_hr_x0 = latents_hr_x0_new

                if (pad1 > 0) and (pad2 > 0) and next_t_lr is not None:
                    lt_down_hr_x0, _ = self._resize_x0_to_x0(latents_hr_x0_new, 1.0 / hr_scale, "", ori_scale=hr_scale)
                    latents_lr = self._synthesize_latents_from_x0(
                        lt_down_hr_x0,
                        latents_lr_new,
                        stage2_lr_noise,
                        bd1_mask,
                        next_t_lr,
                    )
                else:
                    latents_lr = latents_lr_new
                    latents_lr_x0 = latents_lr_x0_new

                progress_bar.update()

        self._current_timestep = None
        
        # clear mask cache in DiT
        self.transformer._mixed_mask_cache = None

        # fill: lr with hr | hr with lr
        lt_lr_fake, _ = self._resize_x0_to_x0(latents_hr, 1./hr_scale, "repeat", ori_scale=hr_scale, align_mean=False)
        latents_lr = latents_lr * actual_lr_mask + lt_lr_fake * (1. - actual_lr_mask)
        lt_hr_fake, _ = self._resize_x0_to_x0(latents_lr, hr_scale, "repeat", ori_scale=1, align_mean=False)
        latents_hr = latents_hr * actual_hr_mask + lt_hr_fake * (1. - actual_hr_mask)

        # apply distance feather on mask
        dec_lr = self._vae1_decode(latents_lr, hw_scale=1, need_unpack=True)
        dec_lr_up = F.interpolate(dec_lr, scale_factor=hr_scale, mode="bicubic", align_corners=False).clamp_(-1.0, 1.0) 
        dec_hr = self._vae1_decode(latents_hr, hw_scale=2, need_unpack=True) # (1, 3, 1024, 1024)
        core_hr_mask_pix = transform_base_mask(core_hr_mask_LR, hr_scale, patch_size=16) 
        core_hr_mask_pix_feather = feather_mask_from_bool(core_hr_mask_pix, radius=feather_radius)
        image = dec_lr_up.float() * (1. - core_hr_mask_pix_feather) + dec_hr.float() * core_hr_mask_pix_feather

        image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return FluxPipelineOutput(images=image)
