"""RALU ported to FLUX.2-klein, 2-stage foveated variant.

Structure follows pipeline_flux_RALU.py as closely as the base model allows. The only
methodological change is the one the comparison requires: stage 2 upsamples the tokens
selected by a supplied foveation mask instead of the top-k Canny edge blocks, and their
stage 3 (which upsamples everything to uniform full resolution) is dropped so the output
stays mixed-resolution like ours.

Everything else is theirs: the uniform low-res stage 1, the per-stage tailored shift
schedule, NT-DM, the correlated re-noising at the transition, nearest-neighbour token
upsampling, and the +0.5 RoPE id refinement.

Port-only differences from their FLUX.1-dev code, none of them methodological:
  * 4-column (T,H,W,L) position ids instead of 3-column (idx, H, W).
  * latent_dim 128 (patchified 32-ch VAE latents) instead of 64.
  * h_ori = exp(compute_empirical_mu(seq_len, steps)) instead of the fixed FLUX.1-dev
    shift of 3; FLUX.2's exponential-mu shift is the same functional family.
  * NT-DM shift bound widened (their 10.0 is calibrated to h_ori=3 and pins at h_ori=7.56).
  * Correlated noise sampled analytically rather than via a dense MultivariateNormal.
  * diffusers 0.37 batch-norm latent denormalisation before VAE decode.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    Flux2KleinPipeline,
    compute_empirical_mu,
)
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput

from RALU_NTDM_general import NT_DM


class Flux2KleinRALUFovPipeline(Flux2KleinPipeline):
    # ------------------------------------------------------------------
    # Parameters (their set_params, K-stage, no hardcoded level constants)
    # ------------------------------------------------------------------
    def set_params(self, N, e, h_ori=None, shift=None, Z=None, seq_len=4096, f0=2):
        """N: steps per stage. e: stage end timesteps, last must be 1.0."""
        assert len(N) == len(e), "N and e must have the same length"
        assert abs(e[-1] - 1.0) < 1e-12, "last entry of e must be 1.0"

        if h_ori is None:
            # FLUX.2's shift is exp(mu); same family as FLUX.1's shift factor.
            h_ori = math.exp(compute_empirical_mu(image_seq_len=seq_len, num_steps=sum(N)))

        if shift is None or Z is None:
            shift, Z = NT_DM(N, e, h_ori=h_ori)

        s = [0.0]
        alphas = []
        betas = []
        for i in range(len(N) - 1):
            s.append(e[i] / (Z * (1 - e[i]) + e[i]))
            alphas.append(1 - s[i + 1])
            betas.append(1 / (Z * (1 - e[i]) + e[i]))

        self.params = {
            "N": N,
            "shift": list(shift),
            "e": list(e),
            "Z": float(Z),
            "h_ori": float(h_ori),
            "s": s,
            "alphas": alphas,
            "betas": betas,
            "f0": f0,
        }
        return self.params

    id_frame = "coarse"   # "coarse" = RALU-faithful stage-2 frame; "full" = their stage-3 rescale

    def set_foveation_mask(self, mask_coarse: torch.Tensor):
        """mask_coarse: [h_d, w_d] bool/float on the stage-1 (low-res) token grid.
        1 = upsample this block to full resolution in stage 2 (the fovea)."""
        self.fov_mask = mask_coarse.bool().flatten()

    # ------------------------------------------------------------------
    # Correlated noise (their generate_noise, sampled analytically)
    # ------------------------------------------------------------------
    def generate_noise(self, device, height, width, dtype=torch.float32, latent_dim=128, generator=None):
        """One n' per stage transition. Block-correlated: within each upsampled 2x2 group
        the covariance is I + gamma*11^T with gamma = -1/Z^2, matching their Sigma_pixel."""
        f0 = self.params["f0"]
        Z = self.params["Z"]

        M = (height // 16) * (width // 16)
        M_down = M // (f0 ** 2)
        m = int(self.fov_mask.sum().item())          # blocks upsampled (was int(M_down*up_ratio))
        a_1x = M_down - m                            # untouched low-res tokens, block_size 1

        gamma = -1.0 / (Z ** 2) + 1e-6
        b = f0 * f0
        # Sigma = I + gamma*J  is realised by x = z + a*(sum z)*1 with 2a + a^2*b = gamma.
        # Real iff 1 + b*gamma >= 0, i.e. Z >= f0^2 -- guaranteed by NT-DM's Z >= 2 bound.
        disc = 1.0 + b * gamma
        if disc < 0:
            raise ValueError(f"Z={Z} too small for block size {b}: correlated noise undefined")
        a_coef = (-1.0 + math.sqrt(disc)) / b

        gen_device = generator.device if generator is not None else torch.device(device)

        def sample(n_white, n_blocks, white_first):
            """n_white tokens with Sigma=I plus n_blocks groups of b with Sigma=I+gamma*J,
            concatenated in their configs_stage* order."""
            total = n_white + n_blocks * b
            z = torch.randn((1, total, latent_dim), device=gen_device,
                            dtype=torch.float32, generator=generator).to(device)
            if white_first:
                white, blk = z[:, :n_white, :], z[:, n_white:, :]
            else:
                blk, white = z[:, : n_blocks * b, :], z[:, n_blocks * b:, :]
            if n_blocks:
                blk = blk.reshape(1, n_blocks, b, latent_dim)
                blk = (blk + a_coef * blk.sum(dim=2, keepdim=True)).reshape(1, n_blocks * b, latent_dim)
            parts = [white, blk] if white_first else [blk, white]
            return torch.cat(parts, dim=1).to(dtype)

        # transition into stage 2: [1x tokens (white)] ++ [m upsampled foveal blocks]
        n_primes = [sample(n_white=a_1x, n_blocks=m, white_first=True)]

        # transition into stage 3: [newly-upsampled periphery blocks] ++ [foveal tokens (white)]
        if len(self.params["N"]) == 3:
            n_primes.append(sample(n_white=b * m, n_blocks=a_1x, white_first=False))

        self.n_primes = n_primes
        return self.n_primes

    # ------------------------------------------------------------------
    # Their id helpers, widened to 4-column (T,H,W,L) ids
    # ------------------------------------------------------------------
    @staticmethod
    def _upsample_latent_ids(latent_ids, offset):
        """Each coarse token -> 4 sub-tokens at (+0, +offset) in H and W.
        Sub-token order matches the 4x nearest replication along the sequence."""
        # float32, as in their _upsample_latent_ids: diffusers' _prepare_latent_ids returns
        # int64 ids, and inheriting that dtype truncates the 0.5 sub-token offset to 0,
        # collapsing all four sub-tokens of a block onto the same position.
        latent_ids = latent_ids.float()
        offsets = torch.tensor([0.0, offset], dtype=torch.float32, device=latent_ids.device)
        grid_h, grid_w = torch.meshgrid(offsets, offsets, indexing="ij")
        cand = torch.stack([grid_h, grid_w], dim=-1)                     # [2,2,2]

        t_col = latent_ids[:, 0:1]
        hw = latent_ids[:, 1:3]
        l_col = latent_ids[:, 3:4]

        hw_new = hw.unsqueeze(1).unsqueeze(1) + cand.unsqueeze(0)        # [n,2,2,2]
        t_new = t_col.unsqueeze(1).unsqueeze(1).expand(-1, 2, 2, -1)
        l_new = l_col.unsqueeze(1).unsqueeze(1).expand(-1, 2, 2, -1)

        out = torch.cat([t_new, hw_new, l_new], dim=-1)                  # [n,2,2,4]
        return out.reshape(-1, 4)

    @staticmethod
    def _sort_latents_and_ids(latents, latent_ids):
        """Their _sort_latents_and_ids: sort by W then stable-sort by H -> raster order.
        Column indices 1/2 are H/W in the 4-column layout too, so this carries over."""
        _, indices_col2 = torch.sort(latent_ids[:, 2], stable=True)
        latent_ids_sorted_by_col2 = latent_ids[indices_col2]
        _, indices_col1 = torch.sort(latent_ids_sorted_by_col2[:, 1], stable=True)
        final_indices = indices_col2[indices_col1]
        return latents[:, final_indices, :], latent_ids[final_indices]

    @staticmethod
    def _coarse_latent_ids(h, w, device, dtype):
        t = torch.arange(1)
        hh = torch.arange(h)
        ww = torch.arange(w)
        l = torch.arange(1)
        ids = torch.cartesian_prod(t, hh, ww, l)
        # always float: sub-token refinement adds fractional offsets before rescaling
        return ids.to(device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # RALU-native mixed-resolution decode
    # ------------------------------------------------------------------
    def _scatter_mixed_to_full(self, latents, latent_ids, h, w, n_1x, ids_full_res):
        """Place stage-2 tokens on the full-res h x w grid. Low-res tokens are
        nearest-upsampled into their 2x2 slot -- the same F.interpolate(mode='nearest')
        their stage 3 applies, minus the extra denoising.

        In the "full" id frame the ids are already full-resolution integers; in the
        (RALU-faithful) "coarse" frame they are in coarse units and are scaled here
        instead. Either way a low-res token lands on the top-left of its 2x2 block.
        """
        B, _, D = latents.shape
        full = torch.zeros((B, h, w, D), device=latents.device, dtype=latents.dtype)

        scale = 1 if ids_full_res else 2
        ids_1x = latent_ids[:n_1x]
        ids_2x = latent_ids[n_1x:]

        r = (ids_1x[:, 1] * scale).round().long()
        c = (ids_1x[:, 2] * scale).round().long()
        r2 = (ids_2x[:, 1] * scale).round().long()
        c2 = (ids_2x[:, 2] * scale).round().long()
        for name, rr, cc in (("low-res", r, c), ("high-res", r2, c2)):
            if rr.numel() and (rr.max() >= h or cc.max() >= w):
                raise ValueError(
                    f"{name} token ids out of range: max=({int(rr.max())},{int(cc.max())}) "
                    f"grid=({h},{w}) -- position id scaling is wrong"
                )

        for dy in range(2):
            for dx in range(2):
                full[:, r + dy, c + dx, :] = latents[:, :n_1x, :]
        full[:, r2, c2, :] = latents[:, n_1x:, :]

        return full.permute(0, 3, 1, 2)                                   # [B, D, h, w]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[dict] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple = (9, 18, 27),
    ):
        N, shift, e, s, alphas, betas, f0 = (
            self.params[k] for k in ("N", "shift", "e", "s", "alphas", "betas", "f0")
        )
        assert len(N) in (2, 3), (
            "supported schedules: 2-stage (ends mixed-resolution) or 3-stage "
            "(their full design, ends uniform full-resolution)"
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        device = self._execution_device
        batch_size = 1 if isinstance(prompt, str) or prompt is None else len(prompt)

        start_time = time.time()

        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )
        latent_ids = latent_ids[0]                                        # [M, 4]

        h_full, w_full = height // 16, width // 16
        h_d, w_d = h_full // f0, w_full // f0
        B = latents.shape[0]
        D = latents.shape[-1]

        n_1x = None
        for idx in range(len(N)):
            S = 1 - s[idx]
            E = 1 - e[idx]
            sh = shift[idx]
            s_prime = float(S / (sh - (sh - 1) * S))
            e_prime = float(E / (sh - (sh - 1) * E))
            sigmas = np.linspace(s_prime, e_prime, N[idx] + 1)
            timesteps = sh * sigmas / (1 + (sh - 1) * sigmas)
            timesteps = torch.from_numpy(timesteps * 1000).to(device).to(torch.float32)

            # ---- b) up/downsampling -------------------------------------------------
            if idx == 0:
                # Stage 1: uniform low-res. Downsample in VAE-latent space, as they do.
                lat = latents.view(B, h_full, w_full, D).permute(0, 3, 1, 2)
                lat = self._unpatchify_latents(lat)                       # [B, 32, 2h, 2w]
                lat = F.interpolate(lat, size=(lat.shape[-2] // f0, lat.shape[-1] // f0), mode="nearest")
                lat = self._patchify_latents(lat)                         # [B, 128, h_d, w_d]
                latents = self._pack_latents(lat)                         # [B, M_down, 128]
                latent_ids = self._coarse_latent_ids(h_d, w_d, device, latent_ids.dtype)
            elif idx == 1:
                # Stage 2: upsample the foveal blocks (their top-k slot, mask-driven).
                mask = self.fov_mask.to(latents.device)
                indices_all = torch.arange(latents.size(1), device=latents.device)
                indices_2x = indices_all[mask]
                indices_1x = indices_all[~mask]
                n_1x = indices_1x.numel()

                latents_1x = latents[:, indices_1x, :]
                latents_2x = latents[:, indices_2x, :]
                ids_1x = latent_ids[indices_1x, :]
                ids_2x = latent_ids[indices_2x, :]

                if latents_2x.size(1) > 0:
                    latents_2x = F.interpolate(
                        latents_2x.transpose(1, 2).float(),
                        size=latents_2x.size(1) * f0 * f0,
                        mode="nearest",
                    ).transpose(1, 2).to(latents.dtype)
                    ids_2x = self._upsample_latent_ids(ids_2x, 0.5)

                latents = torch.cat([latents_1x, latents_2x], dim=1)
                latent_ids = torch.cat([ids_1x, ids_2x], dim=0)
                if self.id_frame == "full":
                    # Their stage-3 rescale, applied early. Correct for the upsampled
                    # (foveal) tokens, but it leaves the untouched low-res tokens at
                    # spacing 2 in a full-extent frame, i.e. spatially sparse.
                    latent_ids = latent_ids * f0

                # ---- c) add noise ---------------------------------------------------
                alpha, beta = alphas[idx - 1], betas[idx - 1]
                latents = beta * latents + alpha * self.n_primes[idx - 1].to(device, latents.dtype)

            else:
                # Stage 3, verbatim from their idx == 2 branch: upsample the tokens that
                # are still low-res (here the periphery, there everything outside the
                # top-k), rescale all ids to full-resolution coords, re-noise, sort.
                latents_1x = latents[:, :n_1x, :]
                latents_2x = latents[:, n_1x:, :]
                ids_1x = latent_ids[:n_1x, :]
                ids_2x = latent_ids[n_1x:, :]

                if latents_1x.size(1) > 0:
                    latents_1x = F.interpolate(
                        latents_1x.transpose(1, 2).float(),
                        size=latents_1x.size(1) * (f0 ** 2),
                        mode="nearest",
                    ).transpose(1, 2).to(latents.dtype)
                    ids_1x = self._upsample_latent_ids(ids_1x, 0.5)

                latents = torch.cat([latents_1x, latents_2x], dim=1)
                latent_ids = torch.cat([ids_1x, ids_2x], dim=0) * f0

                alpha, beta = alphas[idx - 1], betas[idx - 1]
                latents = beta * latents + alpha * self.n_primes[idx - 1].to(device, latents.dtype)
                latents, latent_ids = self._sort_latents_and_ids(latents, latent_ids)
                n_1x = 0            # everything is at full resolution now

            # ---- d) denoising loop ---------------------------------------------------
            img_ids = latent_ids.unsqueeze(0).expand(B, -1, -1)
            with self.progress_bar(total=N[idx]) as progress_bar:
                for i in range(N[idx]):
                    t = timesteps[i]
                    self._current_timestep = t
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    noise_pred = self.transformer(
                        hidden_states=latents.to(self.transformer.dtype),
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=img_ids,
                        joint_attention_kwargs=self._attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                    if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                        neg = self.transformer(
                            hidden_states=latents.to(self.transformer.dtype),
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=img_ids,
                            joint_attention_kwargs=self._attention_kwargs,
                            return_dict=False,
                        )[0][:, : latents.size(1)]
                        noise_pred = neg + guidance_scale * (noise_pred - neg)

                    dt = (timesteps[i + 1] - timesteps[i]).to(torch.float32) * 1e-3
                    latents = (latents.to(torch.float32) + noise_pred.to(torch.float32) * dt).to(latents.dtype)
                    progress_bar.update()

        self._current_timestep = None
        denoise_time = time.time() - start_time

        # ---- RALU-native mixed-resolution decode ------------------------------------
        ids_full_res = (len(N) == 3) or (self.id_frame == "full")
        latents = self._scatter_mixed_to_full(latents, latent_ids, h_full, w_full, n_1x, ids_full_res)

        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * bn_std + bn_mean
        latents = self._unpatchify_latents(latents)

        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.denoise_time = denoise_time
        self.total_time = time.time() - start_time
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return Flux2PipelineOutput(images=image)
