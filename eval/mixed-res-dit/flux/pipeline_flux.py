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

from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Union

from einops import rearrange
import numpy as np

import torch
import torch.nn.functional as F
from diffusers.utils import is_torch_xla_available

from .pipeline_flux_base import (
    FluxPipeline as FluxPipeline_BASE,
    FluxPipelineOutput,
    PipelineImageInput,
)
from .utils.NTDM import NT_DM
from .utils.helpers import transform_base_mask


class FluxPipeline(FluxPipeline_BASE):
    def set_3stg_params(self, N=[6, 7, 5], e=[0.37, 0.6, 1.0]): 
        assert len(N) == 3 and len(e) == 3
        shift, Z = NT_DM(N, e, self.scheduler.shift)

        s = [0.0]
        alphas = []
        betas = [] 

        for i in range(len(N) - 1):
            s.append(e[i] / (Z * (1 - e[i]) + e[i]))
            alphas.append(1 - s[i + 1])
            betas.append(1 / (Z * (1 - e[i]) + e[i]))

        self.params_3stg = {
            "N": N,
            "shift": shift,
            "e": e,
            "Z": Z, 
            "s": s,
            "alphas": alphas,
            "betas": betas, 
        }

    def _vae1_decode(self, latents, hw_scale=1, need_unpack=True) -> torch.Tensor:
        if need_unpack:
            latents = self._unpack_latents(latents, self._h * hw_scale, self._w * hw_scale, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        return image

    def _tvae_decode(self, latents, hw_scale=1, need_unpack=True) -> torch.Tensor:
        if need_unpack:
            latents = self._unpack_latents(latents, self._h * hw_scale, self._w * hw_scale, self.vae_scale_factor)
        image = self.tvae.decode(latents, return_dict=False)[0]
        return image

    def _pack_lt_w_scale(self, latents, hw_scale) -> torch.Tensor:
        lt_h = 2 * (int(self._h * hw_scale) // (self.vae_scale_factor * 2))
        lt_w = 2 * (int(self._w * hw_scale) // (self.vae_scale_factor * 2))
        latents = self._pack_latents(latents, latents.shape[0], self._c_lt, lt_h, lt_w)
        return latents

    def _resize_x0_to_x0(
        self,
        latents_x0: torch.Tensor,
        resize_scale: int,
        mode: str = "repeat",
        ori_scale: int = 1,
        dec_image: bool = False,
    ) -> torch.Tensor:
        new_scale = resize_scale * ori_scale

        assert mode == "repeat"
        lt = self._unpack_latents(latents_x0, self._h * ori_scale, self._w * ori_scale, self.vae_scale_factor)
        if resize_scale == 2:
            new_lt = lt.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        else:
            new_lt = lt[..., ::2, ::2]
        latents_new_x0 = self._pack_lt_w_scale(new_lt, new_scale)

        image = self._tvae_decode(latents_x0, ori_scale) if dec_image else None
        return latents_new_x0, image

    def _get_saliency(
        self,
        img_lr: torch.Tensor,
    ):
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

    def _step_and_get_x0(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = latents.dtype
        latents_x0 = (
            latents.to(torch.float32) - noise_pred.to(torch.float32) * timestep.to(torch.float32) * 1e-3
        ).to(dtype)
        latents_next = (
            latents.to(torch.float32)
            + noise_pred.to(torch.float32) * (next_timestep - timestep).to(torch.float32) * 1e-3
        ).to(dtype)
        return latents_next, latents_x0

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
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        generator2: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
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
        hr_ratio: float = 0.25,
        hr_scale: int = 2,
    ):
        if num_images_per_prompt != 1:
            raise ValueError("`num_images_per_prompt` must be 1 in this pipeline.")
        assert callback_on_step_end is None

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

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
        self._h, self._w = height, width

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

        latents_lr = None
        latents_hr = None
        latents_hr_noise = None
        latents_lr_x0 = None
        core_mask = None

        latents_lr_noise, latent_lr_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents_lr,
        )

        latents_lr_noise2, _ = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator2,
            latents_lr,
        )

        latents_hr_noise, latent_hr_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height * hr_scale,
            width * hr_scale,
            prompt_embeds.dtype,
            device,
            generator,
            latents_hr,
        )
        latents_hr_noise2, _ = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height * hr_scale,
            width * hr_scale,
            prompt_embeds.dtype,
            device,
            generator2,
            latents_hr,
        )

        latents_lr = latents_lr_noise + 0.0
        bsz = latents_lr.shape[0]
        assert num_images_per_prompt == 1
        p_h, p_w = 2, 2
        ht = (height // self.vae_scale_factor) // p_h
        wt = (width // self.vae_scale_factor) // p_w
        assert ht * wt == latents_lr.shape[1]

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(bsz)
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

        N, shift, e, s, alphas, betas = itemgetter("N", "shift", "e", "s", "alphas", "betas")(self.params_3stg)

        self.scheduler.set_begin_index(0)
        progress_bar = self.progress_bar(total=sum(N))
        for idx in range(len(N)):
            S = 1 - s[idx]
            E = 1 - e[idx]
            s_prime = float(S / (shift[idx] - (shift[idx] - 1) * S))
            e_prime = float(E / (shift[idx] - (shift[idx] - 1) * E))
            sigmas = np.linspace(s_prime, e_prime, N[idx] + 1)
            timesteps = shift[idx] * sigmas / (1 + (shift[idx] - 1) * sigmas)
            timesteps = torch.from_numpy(timesteps * 1000).to(device).to(torch.float32)
            end = N[idx]

            num_inference_steps = len(timesteps)
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

            for i in range(end):
                if self.interrupt:
                    continue

                t = timesteps[i]
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                timestep = t.expand(bsz).to(latents_lr_noise.dtype)

                if idx == 2:
                    if i == 0:
                        if core_mask is not None:
                            _hr_mask = transform_base_mask(core_mask, hr_scale, patch_size=1)
                            _hr_mask = rearrange(_hr_mask, "b c h w -> b (h w) c")
                        else:
                            _hr_mask = None
                        assert _hr_mask is not None
                        _hr_mask = _hr_mask.to(latents_hr.dtype)

                        latents_fake_hr, _ = self._resize_x0_to_x0(latents_lr, hr_scale, "repeat", ori_scale=1)
                        latents_hr = latents_fake_hr * (1.0 - _hr_mask) + latents_hr * _hr_mask

                        alpha, beta = alphas[idx - 1], betas[idx - 1]
                        latents_hr = beta * latents_hr + alpha * latents_hr_noise2

                        latents_lr = None
                        latents_lr_x0 = None

                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=latents_hr,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_hr_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]

                    latents_hr, latents_hr_x0 = self._step_and_get_x0(
                        latents_hr, noise_pred, timesteps[i], timesteps[i + 1]
                    )

                elif idx == 0:
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

                    latents_lr, latents_lr_x0 = self._step_and_get_x0(
                        latents_lr, noise_pred, timesteps[i], timesteps[i + 1]
                    )

                    if i == end - 1:
                        if latents_hr_noise is None:
                            raise ValueError("`latents_hr_noise` must be initialized.")

                        latents_hr_x0, stage1_img_lr = self._resize_x0_to_x0(
                            latents_lr_x0, hr_scale, mode="repeat", ori_scale=1, dec_image=True
                        )
                        latents_hr_x0 = latents_hr_x0.to(latents_lr.dtype)

                        saliency_map, stage1_img_lr_uint8 = self._get_saliency(stage1_img_lr)
                        imp_flat = saliency_map.view(bsz, -1)
                        _, imp_idx = torch.topk(imp_flat, int(hr_ratio * imp_flat.shape[1]), dim=1, largest=True)
                        imp_mask = torch.zeros_like(imp_flat, dtype=torch.bool)
                        imp_mask.scatter_(1, imp_idx, True)
                        core_mask = imp_mask.view(*saliency_map.shape)
                        rev_core_mask = ~core_mask

                else:
                    if i == 0:
                        alpha, beta = alphas[idx - 1], betas[idx - 1]
                        latents_hr, _ = self._resize_x0_to_x0(latents_lr, hr_scale, "repeat", ori_scale=1)
                        latents_hr = beta * latents_hr + alpha * latents_hr_noise
                        latents_lr = beta * latents_lr + alpha * latents_lr_noise2

                    assert i + 1 < len(timesteps)

                    with self.transformer.cache_context("cond"):
                        noise_pred_lr, noise_pred_hr = self.transformer.forward_mixed(
                            hidden_states_lr=latents_lr,
                            hidden_states_hr=latents_hr,
                            mask_lr_tokens=core_mask,
                            mask_lr_keep_tokens=rev_core_mask,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids_lr=latent_lr_image_ids,
                            img_ids_hr=latent_hr_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                        )

                    latents_lr_new, latents_lr_x0 = self._step_and_get_x0(
                        latents_lr, noise_pred_lr, timesteps[i], timesteps[i + 1]
                    )
                    latents_hr_new, latents_hr_x0 = self._step_and_get_x0(
                        latents_hr, noise_pred_hr, timesteps[i], timesteps[i + 1]
                    )

                    latents_lr = latents_lr_new
                    latents_hr = latents_hr_new

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None
        self.transformer._mixed_mask_cache = None

        assert output_type != "latent"

        if latents_lr is not None:
            latents_lr = self._unpack_latents(latents_lr, height, width, self.vae_scale_factor)
        if latents_hr is not None:
            latents_hr = self._unpack_latents(latents_hr, height * hr_scale, width * hr_scale, self.vae_scale_factor)

        image = self._vae1_decode(latents_hr, need_unpack=False)
        image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
