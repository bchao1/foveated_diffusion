"""RALU ported to DiffSynth's FLUX.2 pipeline (2- or 3-stage, foveated mask).

Same method as pipeline_flux2_klein_RALU_fov.py, but built on DiffSynth's
Flux2ImagePipeline instead of diffusers so that RALU and our foveated diffusion share
*exactly* the same framework: same Qwen3 text-encoder path, same VAE wrapper, same noise
generator, same native LoRA loading (no key renaming), same latent conventions.

It subclasses the plain Flux2ImagePipeline, NOT Flux2FoveatedImagePipeline: RALU must not
inherit our CRPA attention or Token AE. The only thing shared with our method is the
foveation mask geometry, so both foveate an identical region.

Method content is RALU's: uniform low-res stage 1, per-stage tailored shift schedule from
NT-DM, correlated re-noising at each transition, nearest-neighbour token upsampling, +0.5
RoPE id refinement, and (3-stage only) their final rescale + raster sort.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from diffsynth.pipelines.flux2_image import Flux2ImagePipeline
from diffsynth.diffusion.flow_match import FlowMatchScheduler

from RALU_NTDM_general import NT_DM


# Published RALU operating points (pipeline_flux_RALU.py:46-65). We keep their stage
# boundaries `e` fixed and rescale only the per-stage step counts, so the schedule's shape
# is theirs and only the total budget is matched to ours.
RALU_PRESETS = {
    # N/e plus the shift/Z constants they publish alongside them. Those constants were
    # solved by NT-DM against FLUX.1-dev's h_ori=3 and are tied to the paired N.
    4: {"N": [5, 6, 7], "e": [0.3, 0.45, 1.0],
        "shift": [5.019, 2.589, 2.226], "Z": 6.317, "up_ratio": 0.3},
    7: {"N": [2, 3, 5], "e": [0.2, 0.3, 1.0],
        "shift": [8.141, 2.856, 2.193], "Z": 6.261, "up_ratio": 0.1},
}


def scale_stage_steps(base_N, total_steps):
    """Rescale per-stage step counts to `total_steps` while preserving their proportions.

    Largest-remainder allocation, so the result sums to total_steps exactly and each stage
    keeps at least one step. E.g. [5,6,7] at 50 -> [14,17,19].
    """
    if total_steps < len(base_N):
        raise ValueError(f"total_steps={total_steps} < number of stages {len(base_N)}")
    denom = sum(base_N)
    exact = [total_steps * n / denom for n in base_N]
    alloc = [max(1, int(math.floor(x))) for x in exact]

    # distribute what's left (or claw back) by largest fractional part
    order = sorted(range(len(alloc)), key=lambda i: exact[i] - math.floor(exact[i]), reverse=True)
    diff = total_steps - sum(alloc)
    i = 0
    while diff > 0:
        alloc[order[i % len(alloc)]] += 1
        diff -= 1
        i += 1
    i = 0
    while diff < 0:
        j = order[-(1 + i % len(alloc))]
        if alloc[j] > 1:
            alloc[j] -= 1
            diff += 1
        i += 1
    return alloc


def preset_for_stages(level, num_stages):
    """Their published preset for `num_stages`.

    They only publish 3-stage schedules. The 2-stage variant is derived by merging their
    last two stages, which keeps their first boundary and their step proportions rather
    than inventing new numbers: N=[5, 6+7], e=[0.3, 1.0].
    """
    p = RALU_PRESETS[level]
    if num_stages == 3:
        return list(p["N"]), list(p["e"])
    if num_stages == 2:
        return [p["N"][0], p["N"][1] + p["N"][2]], [p["e"][0], 1.0]
    raise ValueError(f"num_stages must be 2 or 3, got {num_stages}")


def stage_token_counts(height, width, f0, n_hr_blocks, num_stages):
    """Tokens processed per denoising step in each RALU stage.

    stage 1 uniform low-res, stage 2 mixed (identical to our method's token set),
    stage 3 uniform full-res.
    """
    M = (height // 16) * (width // 16)
    M_down = M // (f0 ** 2)
    mixed = M_down + (f0 ** 2 - 1) * n_hr_blocks
    return [M_down, mixed] if num_stages == 2 else [M_down, mixed, M]


def ours_token_cost(height, width, f0, n_hr_blocks, steps):
    """Our method runs the mixed-resolution token set for every step."""
    M = (height // 16) * (width // 16)
    M_down = M // (f0 ** 2)
    return steps * (M_down + (f0 ** 2 - 1) * n_hr_blocks)


def match_compute_steps(base_N, tokens, target_cost):
    """Total step count whose RALU cost best matches `target_cost`, with the stage
    proportions of `base_N` held fixed.

    Cost is token-steps (sum over stages of steps_i * tokens_i). This is a linear proxy:
    it ignores attention's quadratic term, which is a modest fraction at these sequence
    lengths but does favour RALU's short full-res stage slightly. Check wall-clock too.
    """
    fracs = [n / sum(base_N) for n in base_N]
    per_step = sum(f * t for f, t in zip(fracs, tokens))
    s_float = target_cost / per_step

    best, best_err = None, None
    for s in {max(len(base_N), int(math.floor(s_float))), int(math.ceil(s_float))}:
        if s < len(base_N):
            continue
        alloc = scale_stage_steps(base_N, s)
        cost = sum(n * t for n, t in zip(alloc, tokens))
        err = abs(cost - target_cost)
        if best_err is None or err < best_err:
            best, best_err = (s, alloc, cost), err
    return best   # (total_steps, N, cost)


def ralu_schedule(total_steps=50, level=4, num_stages=3, N=None, e=None,
                  match_compute=False, height=1024, width=1024, f0=2,
                  n_hr_blocks=None, ours_steps=50):
    """Return (N, e, info).

    Default: their preset for `num_stages`, e untouched, N rescaled to `total_steps`.
    With match_compute=True, `total_steps` is instead solved so RALU's token-steps equal
    our method's at `ours_steps` for the same mask. Explicit N/e always override.
    """
    base_N, base_e = preset_for_stages(level, num_stages)
    e_out = list(e) if e is not None else base_e
    info = {"match_compute": bool(match_compute)}

    if N is not None:
        N_out = list(N)
    elif match_compute:
        if n_hr_blocks is None:
            raise ValueError("match_compute requires n_hr_blocks")
        tokens = stage_token_counts(height, width, f0, n_hr_blocks, num_stages)
        target = ours_token_cost(height, width, f0, n_hr_blocks, ours_steps)
        total, N_out, cost = match_compute_steps(base_N, tokens, target)
        info.update(tokens_per_stage=tokens, target_cost=target, ralu_cost=cost,
                    matched_total_steps=total,
                    cost_ratio=round(cost / target, 4))
    else:
        N_out = scale_stage_steps(base_N, total_steps)

    return N_out, e_out, info


class Flux2DiffSynthRALUFovPipeline(Flux2ImagePipeline):
    id_frame = "coarse"   # 2-stage only: "coarse" = RALU stage-2 frame, "full" = early rescale

    @staticmethod
    def from_pretrained(torch_dtype=torch.bfloat16, device="cuda", model_configs=(),
                        tokenizer_config=None, vram_limit=None):
        """Same as Flux2ImagePipeline.from_pretrained, but instantiates this subclass
        (theirs is a staticmethod that hardcodes the base class)."""
        from transformers import AutoTokenizer

        pipe = Flux2DiffSynthRALUFovPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(list(model_configs), vram_limit)
        pipe.text_encoder = model_pool.fetch_model("flux2_text_encoder")
        pipe.text_encoder_qwen3 = model_pool.fetch_model("z_image_text_encoder")
        pipe.dit = model_pool.fetch_model("flux2_dit")
        pipe.vae = model_pool.fetch_model("flux2_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    # ------------------------------------------------------------------
    # FLUX 2x2 latent packing (standard, not specific to either method)
    # ------------------------------------------------------------------
    @staticmethod
    def _patchify_latents(x):
        b, c, h, w = x.shape
        x = x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
        return x.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unpatchify_latents(x):
        b, c, h, w = x.shape
        x = x.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
        return x.reshape(b, c // 4, h * 2, w * 2)

    # ------------------------------------------------------------------
    # Their set_params / NT-DM (K-stage)
    # ------------------------------------------------------------------
    def set_params(self, N, e, h_ori=None, shift=None, Z=None, seq_len=4096, f0=2,
                   z_max=100.0, shift_mode="ntdm", stage_tokens=None):
        """shift_mode:
          'ntdm'    -- per-stage shifts from NT-DM (their design, assumes one global h_ori)
          'natural' -- each stage uses FLUX.2's own shift for its token count and step
                       count, exp(mu(tokens_i, N_i)). FLUX.2's schedule is
                       resolution-dependent, which RALU's single-h_ori model cannot
                       express; this substitutes the shift the model actually expects at
                       each stage's resolution. Z still comes from NT-DM.
        """
        assert len(N) == len(e), "N and e must have the same length"
        assert abs(e[-1] - 1.0) < 1e-12, "last entry of e must be 1.0"
        if h_ori is None:
            # DiffSynth's FLUX.2 schedule is exp(mu); same family as FLUX.1's shift factor.
            h_ori = math.exp(FlowMatchScheduler.compute_empirical_mu(seq_len, sum(N)))
        if shift is None or Z is None:
            shift, Z = NT_DM(N, e, h_ori=h_ori, z_max=z_max)
        if shift_mode == "natural":
            if stage_tokens is None:
                raise ValueError("shift_mode='natural' requires stage_tokens")
            shift = [math.exp(FlowMatchScheduler.compute_empirical_mu(t, n))
                     for t, n in zip(stage_tokens, N)]

        s, alphas, betas = [0.0], [], []
        for i in range(len(N) - 1):
            s.append(e[i] / (Z * (1 - e[i]) + e[i]))
            alphas.append(1 - s[i + 1])
            betas.append(1 / (Z * (1 - e[i]) + e[i]))

        self.params = {
            "N": list(N), "shift": list(shift), "e": list(e), "Z": float(Z),
            "h_ori": float(h_ori), "s": s, "alphas": alphas, "betas": betas, "f0": f0,
        }
        return self.params

    def set_foveation_mask(self, mask_coarse: torch.Tensor):
        """[h_d, w_d] mask on the stage-1 low-res token grid; 1 = foveal block."""
        self.fov_mask = mask_coarse.bool().flatten()

    # ------------------------------------------------------------------
    # Their generate_noise, sampled analytically, via DiffSynth's RNG
    # ------------------------------------------------------------------
    def generate_ralu_noise(self, height, width, seed=None, rand_device="cpu", latent_dim=128):
        f0 = self.params["f0"]
        Z = self.params["Z"]
        M = (height // 16) * (width // 16)
        M_down = M // (f0 ** 2)
        m = int(self.fov_mask.sum().item())
        a_1x = M_down - m
        b = f0 * f0

        gamma = -1.0 / (Z ** 2) + 1e-6
        disc = 1.0 + b * gamma
        if disc < 0:
            raise ValueError(f"Z={Z} too small for block size {b}")
        a_coef = (-1.0 + math.sqrt(disc)) / b

        def sample(n_white, n_blocks, white_first, offset):
            total = n_white + n_blocks * b
            z = self.generate_noise(
                (1, total, latent_dim),
                seed=None if seed is None else seed + offset,
                rand_device=rand_device, rand_torch_dtype=torch.float32,
            ).float()
            if white_first:
                white, blk = z[:, :n_white, :], z[:, n_white:, :]
            else:
                blk, white = z[:, : n_blocks * b, :], z[:, n_blocks * b:, :]
            if n_blocks:
                blk = blk.reshape(1, n_blocks, b, latent_dim)
                blk = (blk + a_coef * blk.sum(dim=2, keepdim=True)).reshape(1, n_blocks * b, latent_dim)
            parts = [white, blk] if white_first else [blk, white]
            return torch.cat(parts, dim=1).to(self.torch_dtype)

        n_primes = [sample(a_1x, m, True, 1)]
        if len(self.params["N"]) == 3:
            n_primes.append(sample(b * m, a_1x, False, 2))
        self.n_primes = n_primes
        return n_primes

    # ------------------------------------------------------------------
    # Their id helpers, on 4-column (T,H,W,L) ids
    # ------------------------------------------------------------------
    @staticmethod
    def _upsample_latent_ids(latent_ids, offset):
        # float32 as in their code: integer ids would truncate the 0.5 offset to 0 and
        # collapse a block's four sub-tokens onto one position.
        latent_ids = latent_ids.float()
        offs = torch.tensor([0.0, offset], dtype=torch.float32, device=latent_ids.device)
        gh, gw = torch.meshgrid(offs, offs, indexing="ij")
        cand = torch.stack([gh, gw], dim=-1)

        t_col, hw, l_col = latent_ids[:, 0:1], latent_ids[:, 1:3], latent_ids[:, 3:4]
        hw_new = hw.unsqueeze(1).unsqueeze(1) + cand.unsqueeze(0)
        t_new = t_col.unsqueeze(1).unsqueeze(1).expand(-1, 2, 2, -1)
        l_new = l_col.unsqueeze(1).unsqueeze(1).expand(-1, 2, 2, -1)
        return torch.cat([t_new, hw_new, l_new], dim=-1).reshape(-1, 4)

    @staticmethod
    def _sort_latents_and_ids(latents, latent_ids):
        _, i2 = torch.sort(latent_ids[:, 2], stable=True)
        _, i1 = torch.sort(latent_ids[i2][:, 1], stable=True)
        final = i2[i1]
        return latents[:, final, :], latent_ids[final]

    @staticmethod
    def _coarse_latent_ids(h, w, device):
        ids = torch.cartesian_prod(torch.arange(1), torch.arange(h), torch.arange(w), torch.arange(1))
        return ids.to(device=device, dtype=torch.float32)

    def _scatter_mixed_to_full(self, latents, latent_ids, h, w, n_1x, ids_full_res):
        """Nearest-upsample the low-res tokens into their 2x2 slot and place the rest
        exactly -- the same F.interpolate(mode='nearest') their stage 3 uses."""
        B, _, D = latents.shape
        full = torch.zeros((B, h, w, D), device=latents.device, dtype=latents.dtype)
        scale = 1 if ids_full_res else 2

        ids_1x, ids_2x = latent_ids[:n_1x], latent_ids[n_1x:]
        r = (ids_1x[:, 1] * scale).round().long()
        c = (ids_1x[:, 2] * scale).round().long()
        r2 = (ids_2x[:, 1] * scale).round().long()
        c2 = (ids_2x[:, 2] * scale).round().long()
        for nm, rr, cc in (("low-res", r, c), ("high-res", r2, c2)):
            if rr.numel() and (rr.max() >= h or cc.max() >= w):
                raise ValueError(f"{nm} ids out of range for {h}x{w}: position scaling is wrong")

        for dy in range(2):
            for dx in range(2):
                full[:, r + dy, c + dx, :] = latents[:, :n_1x, :]
        full[:, r2, c2, :] = latents[:, n_1x:, :]
        return full.permute(0, 3, 1, 2)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        embedded_guidance: float = 4.0,
        height: int = 1024,
        width: int = 1024,
        seed: int = None,
        rand_device: str = "cpu",
        progress_bar_cmd=tqdm,
    ):
        N, shift, e, s, alphas, betas, f0 = (
            self.params[k] for k in ("N", "shift", "e", "s", "alphas", "betas", "f0")
        )
        assert len(N) in (2, 3), "supported: 2-stage (mixed-res out) or 3-stage (full-res out)"

        # Units give us prompt embeddings, initial noise and ids exactly as our runs do.
        self.scheduler.set_timesteps(sum(N), denoising_strength=1.0,
                                     dynamic_shift_len=height // 16 * width // 16)
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale, "embedded_guidance": embedded_guidance,
            "input_image": None, "denoising_strength": 1.0,
            "edit_image": None, "edit_image_auto_resize": True,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": sum(N),
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        self.load_models_to_device(self.in_iteration_models)
        latents = inputs_shared["latents"]
        h_full, w_full = height // 16, width // 16
        h_d, w_d = h_full // f0, w_full // f0
        B, D = latents.shape[0], latents.shape[-1]
        latent_ids = None
        n_1x = None

        start_t = time.time()
        for idx in range(len(N)):
            S, E, sh = 1 - s[idx], 1 - e[idx], shift[idx]
            s_prime = float(S / (sh - (sh - 1) * S))
            e_prime = float(E / (sh - (sh - 1) * E))
            sigmas = np.linspace(s_prime, e_prime, N[idx] + 1)
            timesteps = torch.from_numpy(
                sh * sigmas / (1 + (sh - 1) * sigmas) * 1000
            ).to(self.device).to(torch.float32)

            if idx == 0:
                # Stage 1: uniform low-res, downsampled in VAE-latent space as they do.
                lat = latents.view(B, h_full, w_full, D).permute(0, 3, 1, 2)
                lat = self._unpatchify_latents(lat)
                lat = F.interpolate(lat, size=(lat.shape[-2] // f0, lat.shape[-1] // f0), mode="nearest")
                lat = self._patchify_latents(lat)
                latents = lat.reshape(B, D, h_d * w_d).permute(0, 2, 1)
                latent_ids = self._coarse_latent_ids(h_d, w_d, self.device)

            elif idx == 1:
                # Stage 2: upsample the foveal blocks (their top-k slot, mask-driven).
                mask = self.fov_mask.to(latents.device)
                idx_all = torch.arange(latents.size(1), device=latents.device)
                indices_2x, indices_1x = idx_all[mask], idx_all[~mask]
                n_1x = indices_1x.numel()

                latents_1x, latents_2x = latents[:, indices_1x, :], latents[:, indices_2x, :]
                ids_1x, ids_2x = latent_ids[indices_1x, :], latent_ids[indices_2x, :]

                if latents_2x.size(1) > 0:
                    latents_2x = F.interpolate(
                        latents_2x.transpose(1, 2).float(),
                        size=latents_2x.size(1) * f0 * f0, mode="nearest",
                    ).transpose(1, 2).to(latents.dtype)
                    ids_2x = self._upsample_latent_ids(ids_2x, 0.5)

                latents = torch.cat([latents_1x, latents_2x], dim=1)
                latent_ids = torch.cat([ids_1x, ids_2x], dim=0)
                if len(N) == 2 and self.id_frame == "full":
                    latent_ids = latent_ids * f0

                latents = betas[0] * latents + alphas[0] * self.n_primes[0].to(self.device, latents.dtype)

            else:
                # Stage 3, verbatim from their idx == 2: upsample whatever is still
                # low-res, rescale all ids to full-res coords, re-noise, raster sort.
                latents_1x, latents_2x = latents[:, :n_1x, :], latents[:, n_1x:, :]
                ids_1x, ids_2x = latent_ids[:n_1x, :], latent_ids[n_1x:, :]
                if latents_1x.size(1) > 0:
                    latents_1x = F.interpolate(
                        latents_1x.transpose(1, 2).float(),
                        size=latents_1x.size(1) * (f0 ** 2), mode="nearest",
                    ).transpose(1, 2).to(latents.dtype)
                    ids_1x = self._upsample_latent_ids(ids_1x, 0.5)

                latents = torch.cat([latents_1x, latents_2x], dim=1)
                latent_ids = torch.cat([ids_1x, ids_2x], dim=0) * f0
                latents = betas[1] * latents + alphas[1] * self.n_primes[1].to(self.device, latents.dtype)
                latents, latent_ids = self._sort_latents_and_ids(latents, latent_ids)
                n_1x = 0

            if getattr(self, "debug_trace", False):
                exp = math.sqrt(s[idx] ** 2 + (1 - s[idx]) ** 2)
                print(f"  [stage {idx+1}] entering: std {latents.float().std():.4f} "
                      f"(expected {exp:.4f})  sigma {1-s[idx]:.4f}  tokens {latents.shape[1]}")
            img_ids = latent_ids.unsqueeze(0).expand(B, -1, -1)
            for i in progress_bar_cmd(range(N[idx]), desc=f"Stage {idx+1}"):
                t = timesteps[i].unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                call_shared = dict(inputs_shared)
                call_shared["latents"] = latents
                call_shared["image_ids"] = img_ids
                noise_pred = self.cfg_guided_model_fn(
                    self.model_fn, cfg_scale, call_shared, inputs_posi, inputs_nega,
                    dit=self.dit, timestep=t, progress_id=i,
                )
                dt = (timesteps[i + 1] - timesteps[i]).to(torch.float32) * 1e-3
                latents = (latents.float() + noise_pred.float() * dt).to(latents.dtype)

            if getattr(self, "debug_trace", False):
                exp = math.sqrt(e[idx] ** 2 + (1 - e[idx]) ** 2)
                print(f"  [stage {idx+1}] leaving:  std {latents.float().std():.4f} "
                      f"(expected {exp:.4f})  sigma {1-e[idx]:.4f}")

        vit_end_t = time.time()

        # RALU-native mixed-resolution decode, then DiffSynth's VAE
        self.load_models_to_device(["vae"])
        ids_full_res = (len(N) == 3) or (self.id_frame == "full")
        latents_spatial = self._scatter_mixed_to_full(
            latents, latent_ids, h_full, w_full, n_1x, ids_full_res
        )
        self.final_latents = latents_spatial.detach().float().cpu()   # debug hook
        image = self.vae.decode(latents_spatial)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        self.denoise_time = vit_end_t - start_t
        self.total_time = time.time() - start_t
        return image
