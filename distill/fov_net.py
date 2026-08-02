"""FastGen network wrapper around the DiffSynth foveated FLUX.2 pipeline.

The foveated DiT is a plain flow-matching velocity field on a *reduced* mixed-resolution
token sequence `[B, L, 128]`, where

    L = m + (h*w - m) / f^2      (m = number of high-res tokens, f = lr_downsample_factor)

Because every operation FastGen's distillation methods rely on (`forward_process`,
`flow_to_x0`, the VSD loss) is elementwise / reduces over `range(1, ndim)`, wrapping the
foveated DiT as a `FastGenNetwork` over `[B, L, C]` is sufficient to unlock DMD.

Time convention (verified against both codebases):
    FastGen `rf`:   x_t = (1-t) x_0 + t * eps,  flow = eps - x_0,  x_0 = x_t - t * flow
    DiffSynth FLUX2: z_t = (1-sigma) z_0 + sigma * eps,  target = eps - z_0
    => t (FastGen) == sigma (DiffSynth) == timestep/1000
The DiT itself expects `timestep` in [0, 1] and rescales by 1000 internally, so
`model_fn`-style callers pass `t * 1000`.

Roles (student / critic / teacher) are three views of ONE DiT:
    teacher = base weights (with the foveated SFT LoRA fused in)
    student = base + LoRA adapter "student"
    critic  = base + LoRA adapter "critic"   (the DMD fake score)
This keeps a single 6.1 GB copy of the transformer instead of three.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "DiffSynth-Studio") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "DiffSynth-Studio"))

from peft import LoraConfig, inject_adapter_in_model  # noqa: E402
from peft.tuners.tuners_utils import BaseTunerLayer  # noqa: E402

from diffsynth.core import ModelConfig, load_state_dict  # noqa: E402
from diffsynth.diffusion.flow_match import FlowMatchScheduler  # noqa: E402
from diffsynth.pipelines.flux2_image_foveated import (  # noqa: E402
    Flux2FoveatedImagePipeline,
    Flux2FoveatedUnit_Qwen3PromptEmbedder,
    Flux2FoveatedUnit_PromptEmbedder,
)
from diffsynth.utils.lora.general import GeneralLoRALoader  # noqa: E402

from fastgen.networks.network import FastGenNetwork  # noqa: E402
from fastgen.networks.noise_schedule import NET_PRED_TYPES  # noqa: E402

ROLES = ("student", "critic", "teacher")


# ---------------------------------------------------------------------------
# mask helpers (identical geometry to diffsynth.diffusion.loss / logger)
# ---------------------------------------------------------------------------
def circular_mask(h: int, w: int, center: Tuple[float, float], r: float, device) -> torch.Tensor:
    """center in normalized (-0.5, 0.5) coords, r relative to half-diagonal. 1 = high-res."""
    cx = (center[0] + 0.5) * w
    cy = (center[1] + 0.5) * h
    radius_px = r * ((h**2 + w**2) ** 0.5 / 2.0)
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius_px**2


def random_circular_mask(
    h: int,
    w: int,
    device,
    generator: Optional[torch.Generator] = None,
    center_range: Tuple[float, float] = (-0.3, 0.3),
    r_range: Tuple[float, float] = (0.2, 0.5),
) -> torch.Tensor:
    """Same sampling distribution as `_create_random_foveation_mask` used by the SFT loss."""

    def _u(lo, hi):
        return torch.rand(1, generator=generator).item() * (hi - lo) + lo

    cx = _u(*center_range)
    cy = _u(*center_range)
    r = _u(*r_range)
    return circular_mask(h, w, (cx, cy), r, device)


VALIDATION_POSITIONS = (
    ("left", (-0.3, 0.0)),
    ("right", (0.3, 0.0)),
    ("top", (0.0, -0.3)),
    ("bottom", (0.0, 0.3)),
)
VALIDATION_RADIUS = 0.25  # matches WandbModelLogger.generate_validation_images


def flux2_sigmas(num_steps: int, latent_seq_len: int) -> torch.Tensor:
    """The exact shifted sigma schedule DiffSynth uses at inference for FLUX.2.

    `latent_seq_len` is the FULL-resolution token count (h*w), matching
    `Flux2FoveatedImagePipeline.__call__`, which passes `height//16 * width//16`.
    """
    sched = FlowMatchScheduler("FLUX.2")
    sched.set_timesteps(num_steps, dynamic_shift_len=latent_seq_len)
    return sched.sigmas.clone()


class Flux2FoveatedNet(FastGenNetwork):
    """Foveated FLUX.2 DiT exposed as a FastGen flow-matching network over reduced tokens."""

    def __init__(
        self,
        model_id_with_origin_paths: str,
        tokenizer_path: str,
        lora_path: Optional[str] = None,
        lora_alpha: float = 1.0,
        lora_rank: int = 32,
        lora_target_modules: str = "",
        device: Union[str, torch.device] = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        height: int = 1024,
        width: int = 1024,
        lr_downsample_factor: int = 4,
        embedded_guidance: float = 4.0,
        use_gradient_checkpointing: bool = True,
        inject_adapters: bool = True,
        max_t: float = 0.999,
        min_t: float = 0.001,
    ):
        super().__init__(net_pred_type="flow", schedule_type="rf", max_t=max_t)

        self.torch_dtype = torch_dtype
        self._device = torch.device(device)
        self.height, self.width = height, width
        self.latent_height, self.latent_width = height // 16, width // 16
        self.full_seq_len = self.latent_height * self.latent_width
        self.lr_factor = lr_downsample_factor
        self.embedded_guidance = embedded_guidance
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.t_min, self.t_max = min_t, max_t

        model_configs = [
            _parse_model_config(p) for p in model_id_with_origin_paths.split(",") if p.strip()
        ]
        self.pipe = Flux2FoveatedImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=_parse_model_config(tokenizer_path),
        )
        self.pipe.is_foveated_pipeline = True
        assert self.pipe.dit is not None, "Failed to load the foveated DiT."

        # --- fuse the foveated SFT LoRA into the base weights -> this IS the teacher ---
        self.lora_path = lora_path
        if lora_path:
            sd = load_state_dict(lora_path)
            GeneralLoRALoader(device="cpu", torch_dtype=torch_dtype).fuse_lora_to_base_model(
                self.pipe.dit, sd, alpha=lora_alpha
            )
            print(f"[Flux2FoveatedNet] fused teacher LoRA {lora_path} (alpha={lora_alpha})")
        else:
            print("[Flux2FoveatedNet] WARNING: no --teacher_lora_path; teacher is the *base* "
                  "FLUX.2 model, which has never seen mixed-resolution sequences.")

        self.pipe.eval()
        self.pipe.requires_grad_(False)

        # --- two independent LoRA adapters on the shared frozen base ---
        self.adapters_injected = False
        if inject_adapters:
            targets = (
                lora_target_modules.split(",") if lora_target_modules else _DEFAULT_LORA_TARGETS
            )
            for name in ("student", "critic"):
                cfg = LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=targets)
                self.pipe.dit = inject_adapter_in_model(cfg, self.pipe.dit, adapter_name=name)
            # peft injects in the base dtype; upcast trainable params for stable optimisation
            for n, p in self.pipe.dit.named_parameters():
                if "lora_" in n:
                    p.data = p.data.to(torch_dtype)
            self.adapters_injected = True
            n_lora = sum(p.numel() for n, p in self.pipe.dit.named_parameters() if "lora_" in n)
            print(f"[Flux2FoveatedNet] injected student+critic LoRA (rank={lora_rank}), "
                  f"{n_lora/2/1e6:.2f}M params each")

        # FLUX.2-klein-base-4B is NOT guidance-distilled: its checkpoint has no
        # `guidance_embedder`, so Flux2TimestepGuidanceEmbeddings returns the timestep
        # embedding alone and silently discards `guidance`. Real CFG (two forwards) is the
        # only guidance mechanism available -- see `teacher_cfg_scale`.
        self.has_guidance_embed = (
            getattr(self.pipe.dit.time_guidance_embed, "guidance_embedder", None) is not None
        )
        if not self.has_guidance_embed:
            print("[Flux2FoveatedNet] NOTE: this DiT has no guidance embedder "
                  "(guidance_embeds=False); `embedded_guidance` is inert. Guidance must come "
                  "from CFG (teacher_cfg_scale).")

        self._role = "teacher"
        self._expect_role = None   # set by callers that pin a role, checked in forward()
        self.set_role("teacher")

        self._text_unit_qwen3 = Flux2FoveatedUnit_Qwen3PromptEmbedder()
        self._text_unit_mistral = Flux2FoveatedUnit_PromptEmbedder()

    # ------------------------------------------------------------------ roles
    def set_role(self, role: str) -> None:
        assert role in ROLES, role
        if not self.adapters_injected:
            self._role = "teacher"
            return
        for mod in self.pipe.dit.modules():
            if isinstance(mod, BaseTunerLayer):
                if role == "teacher":
                    mod.enable_adapters(False)
                else:
                    mod.enable_adapters(True)
                    mod.set_adapter(role)
        self._role = role

    @contextlib.contextmanager
    def role(self, role: str):
        prev = self._role
        self.set_role(role)
        try:
            yield self
        finally:
            self.set_role(prev)

    def lora_parameters(self, role: str) -> List[torch.nn.Parameter]:
        return [p for n, p in self.pipe.dit.named_parameters() if f".{role}." in n and "lora_" in n]

    def lora_state_dict(self, role: str) -> dict:
        return {
            n: p.detach().cpu()
            for n, p in self.pipe.dit.named_parameters()
            if f".{role}." in n and "lora_" in n
        }

    # ----------------------------------------------------------------- device
    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        dev, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dev is not None:
            self._device = torch.device(dev)
            self.pipe.device = self._device
        return out

    # ------------------------------------------------------------ conditioning
    @torch.no_grad()
    def encode_prompt(self, prompts: Union[str, Sequence[str]]) -> dict:
        """Text embeddings, using the pipeline's own unit so behaviour matches SFT/inference."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        if self.pipe.text_encoder_qwen3 is not None:
            out = self._text_unit_qwen3.process(self.pipe, prompts)
        else:
            out = self._text_unit_mistral.process(self.pipe, prompts)
        assert out, "No text encoder available in the pipeline."
        return {
            "prompt_embeds": out["prompt_embeds"].to(self.torch_dtype),
            "text_ids": out["text_ids"],
        }

    def make_condition(self, text: dict, mask: torch.Tensor) -> dict:
        """Bundle text + foveation mask. The SAME dict goes to student, critic and teacher."""
        return {
            "prompt_embeds": text["prompt_embeds"],
            "text_ids": text["text_ids"],
            "foveation_mask": mask,
        }

    def sample_mask(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return random_circular_mask(
            self.latent_height, self.latent_width, self.device, generator=generator
        )

    def token_count(self, mask: torch.Tensor) -> int:
        f = self.lr_factor
        h_d, w_d = self.latent_height // f, self.latent_width // f
        blocks = mask.view(h_d, f, w_d, f).permute(0, 2, 1, 3).reshape(h_d, w_d, f * f)
        hr = int((blocks.sum(-1) > 0).sum().item())
        return hr * f * f + (h_d * w_d - hr)

    # --------------------------------------------------------------- latents
    @torch.no_grad()
    def encode_to_fov_latents(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """image [B,3,H,W] in [-1,1] -> clean foveated latents z0_fov [B,L,128].

        Mirrors the SFT target construction exactly: high-res tokens come from E(x),
        peripheral tokens from E(Down(x)), merged by the mask.
        """
        hr = self.pipe.vae.encode(image.to(self.device, self.torch_dtype))
        hr = rearrange(hr, "B C H W -> B (H W) C")
        small = F.interpolate(
            image.to(self.device, self.torch_dtype),
            size=(image.shape[-2] // self.lr_factor, image.shape[-1] // self.lr_factor),
            mode="bicubic",
            align_corners=False,
        )
        lr = self.pipe.vae.encode(small)
        lr = rearrange(lr, "B C H W -> B (H W) C")
        return self.pipe._downsample_latents(
            hr,
            self.latent_height,
            self.latent_width,
            foveation_mask=mask,
            token_downsampler=None,
            latents_downsampled=lr,
            lr_factor=self.lr_factor,
        )

    @torch.no_grad()
    def decode_fov_latents(self, z_fov: torch.Tensor, mask: torch.Tensor) -> List[Image.Image]:
        """`decode_mode="merge"`: decode LR and HR paths separately and blend by the mask."""
        B = z_fov.shape[0]
        f, lh, lw = self.lr_factor, self.latent_height, self.latent_width
        h_d, w_d = lh // f, lw // f

        low = self.pipe._downsample_remaining_high_res(
            z_fov, lh, lw, foveation_mask=mask, lr_factor=f
        )
        low_img = self.pipe.vae.decode(low.view(B, h_d, w_d, -1).permute(0, 3, 1, 2))
        high = self.pipe._upsample_latents(z_fov, lh, lw, foveation_mask=mask, lr_factor=f)
        high_img = self.pipe.vae.decode(high.view(B, lh, lw, -1).permute(0, 3, 1, 2))

        low_img = F.interpolate(
            low_img, size=(self.height, self.width), mode="bicubic", align_corners=False
        )
        m = F.interpolate(
            mask[None, None].float(), size=(self.height, self.width), mode="bicubic"
        ).to(device=low_img.device, dtype=low_img.dtype)
        merged = low_img * (1 - m) + high_img * m
        return [self.pipe.vae_output_to_image(merged[i : i + 1]) for i in range(B)]

    # --------------------------------------------------------------- forward
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        embedded_guidance: Optional[float] = None,
        **fwd_kwargs,
    ) -> torch.Tensor:
        if return_features_early or (feature_indices and len(feature_indices)):
            raise NotImplementedError(
                "Feature extraction (GAN heads) is not supported for mixed-resolution "
                "token sequences; run DMD with gan_loss_weight=0."
            )
        if return_logvar:
            raise NotImplementedError("logvar head not present on the foveated DiT.")

        fwd_pred_type = fwd_pred_type or self.net_pred_type
        assert fwd_pred_type in NET_PRED_TYPES, fwd_pred_type
        if self._expect_role is not None and self._role != self._expect_role:
            raise RuntimeError(
                f"forward() ran under adapter role {self._role!r} but {self._expect_role!r} was "
                "expected. This silently rendered 'teacher' references with the student/critic "
                "adapter once before -- see sample_teacher/sample_student, which now own the role."
            )

        B = x_t.shape[0]
        mask = condition["foveation_mask"]
        ids, res_mask, res_mask_tl = Flux2FoveatedImagePipeline._prepare_latent_image_ids_foveation(
            B,
            self.latent_height,
            self.latent_width,
            x_t.device,
            x_t.dtype,
            mask,
            lr_factor=self.lr_factor,
        )
        if ids.ndim == 2:
            ids = ids.unsqueeze(0).expand(B, -1, -1)

        t = t.to(x_t.device)
        if t.ndim == 0:
            t = t[None]
        if t.shape[0] != B:
            t = t.expand(B)

        guidance = self.embedded_guidance if embedded_guidance is None else embedded_guidance
        guidance_t = torch.full((B,), float(guidance), device=x_t.device, dtype=x_t.dtype)

        prompt_embeds = condition["prompt_embeds"]
        text_ids = condition["text_ids"]
        if prompt_embeds.shape[0] != B:
            prompt_embeds = prompt_embeds.expand(B, -1, -1)
            text_ids = text_ids.expand(B, -1, -1)

        # Reproduce the reference pipeline's cast chain EXACTLY. It builds
        # `timestep_val = (sigma*1000).to(torch_dtype)` (bf16, where the spacing near 1000 is
        # ~2), then `model_fn` divides by 1000, then the DiT multiplies by 1000 again. Casting
        # t in [0,1] to bf16 instead quantises on a 0.0039 grid -- a *different* (in fact
        # finer) discretisation, which compounds over 50 Euler steps into ~0.4/255 pixel drift
        # from the reference, ~3x that under CFG w=4. Matching the chain keeps our sampler and
        # `Flux2FoveatedImagePipeline.__call__` numerically identical.
        timestep_dit = (t.to(torch.float32) * 1000).to(self.torch_dtype) / 1000

        out = self.pipe.dit(
            hidden_states=x_t,
            # the DiT rescales `timestep` by 1000 internally, so hand it t in [0, 1]
            timestep=timestep_dit,
            guidance=guidance_t,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=ids,
            resolution_mask=res_mask,
            resolution_mask_top_left=res_mask_tl,
            lr_factor=self.lr_factor,
            use_gradient_checkpointing=self.use_gradient_checkpointing and torch.is_grad_enabled(),
            use_gradient_checkpointing_offload=False,
        )
        out = out[:, : x_t.shape[1]]
        return self.noise_scheduler.convert_model_output(
            x_t, out, t, src_pred_type="flow", target_pred_type=fwd_pred_type
        )

    # --------------------------------------------------------------- sampling
    def student_t_list(self, num_steps: int, mode: str = "flux2",
                       mu: Optional[float] = None) -> torch.Tensor:
        """Timesteps for the few-step student, descending and terminating at 0.

        `flux2` reuses the native shifted inference schedule. Beware: FLUX.2's empirical
        shift is very top-heavy, so a 4-step native schedule is [1.0, .967, .908, .767] --
        the student would never see an input below t=0.77. `uniform` (the default, and what
        DMD2 does elsewhere) spreads the steps over the whole noise range instead.
        """
        if mode == "flux2":
            if mu is None:
                sig = flux2_sigmas(num_steps, self.full_seq_len).to(torch.float64)
            else:
                # Same map as set_timesteps_flux2, with an explicit shift. The student's shift
                # is a hyperparameter OF THE STUDENT: FLUX.2's compute_empirical_mu is fit
                # between 10 and 200 steps, so its 4-step value (mu=2.29, shift=9.89) is an
                # extrapolation tuned for the base model's few-step behaviour, not for a
                # distilled generator. Sweeping it is legitimate -- but a value other than the
                # pipeline's own gives up drop-in compatibility: inference must then be handed
                # matching sigmas rather than just num_inference_steps=N.
                lin = torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float64)
                e = math.exp(mu)
                sig = e / (e + (1.0 / lin - 1.0))
            t = torch.clamp(sig, max=self.t_max)
        elif mode == "uniform":
            t = torch.linspace(self.t_max, 0.0, num_steps + 1, dtype=torch.float64)[:-1]
        else:
            raise ValueError(f"unknown student_t_mode {mode}")
        return torch.cat([t, torch.zeros(1, dtype=torch.float64)]).to(
            self.noise_scheduler.t_precision
        )

    def _x0_step(self, x, t_scalar, condition, B):
        t_b = t_scalar.expand(B).to(self.noise_scheduler.t_precision)
        return self(x, t_b, condition=condition, fwd_pred_type="x0")

    @torch.no_grad()
    def sample_student(
        self,
        noise: torch.Tensor,
        condition: dict,
        t_list: torch.Tensor,
        sample_type: str = "sde",
        generator: Optional[torch.Generator] = None,
        cfg_scale: float = 1.0,
        neg_condition: Optional[dict] = None,
    ) -> torch.Tensor:
        """Few-step x0-prediction + re-noise sampling (matches FastGen's student loop).

        Pass `generator` for the SDE branch during validation. Without it the intermediate
        re-noising draws from the global RNG, and two validations of the *same* weights differ
        by ~10/255 pixel MAE -- enough to swamp the training signal you are trying to read.
        """
        B = noise.shape[0]
        t_list = t_list.to(noise.device)
        prev_role = self._role
        self.set_role("student")
        self._expect_role = "student"
        x = self.noise_scheduler.latents(noise=noise, t_init=t_list[0])
        use_cfg = cfg_scale != 1.0 and neg_condition is not None
        x_pred = x
        for t_cur, t_next in zip(t_list[:-1], t_list[1:]):
            x_pred = self._x0_step(x, t_cur, condition, B)
            if use_cfg:
                x_neg = self._x0_step(x, t_cur, neg_condition, B)
                x_pred = x_neg + cfg_scale * (x_pred - x_neg)
            if t_next > 0:
                if sample_type == "ode":
                    eps = self.noise_scheduler.x0_to_eps(
                        xt=x, x0=x_pred, t=t_cur.expand(B).to(self.noise_scheduler.t_precision)
                    )
                elif generator is not None:
                    eps = torch.randn(
                        x_pred.shape, generator=generator,
                        device=generator.device, dtype=torch.float32,
                    ).to(device=x_pred.device, dtype=x_pred.dtype)
                else:
                    eps = torch.randn_like(x_pred)
                x = self.noise_scheduler.forward_process(
                    x_pred, eps, t_next.expand(B).to(self.noise_scheduler.t_precision)
                )
        self._expect_role = None
        self.set_role(prev_role)
        return x_pred

    @torch.no_grad()
    def sample_teacher(
        self,
        noise: torch.Tensor,
        condition: dict,
        num_steps: int = 50,
        cfg_scale: float = 1.0,
        neg_condition: Optional[dict] = None,
        sigmas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Euler flow-matching sampling on the reduced sequence, using DiffSynth's exact
        FLUX.2 shifted schedule (the same one `Flux2FoveatedImagePipeline.__call__` uses).

        `cfg_scale > 1` runs true classifier-free guidance (two forwards per step), which is
        the *only* guidance this model has -- it carries no guidance embedding.
        """
        B = noise.shape[0]
        prev_role = self._role
        # frozen base weights -- NOT whatever role training happened to leave active
        self.set_role("teacher")
        self._expect_role = "teacher"
        # `sigmas` override lets the caller integrate on the STUDENT's timestep grid, which is
        # required for an apples-to-apples few-step baseline: at only 4 steps the uniform grid
        # and FLUX.2's shifted grid give very different trajectories, and scoring the student
        # against a different-grid teacher measures grid mismatch, not student error.
        if sigmas is None:
            sigmas = flux2_sigmas(num_steps, self.full_seq_len)
        sigmas = sigmas.to(noise.device)
        num_steps = len(sigmas)
        x = noise * sigmas[0]
        use_cfg = cfg_scale != 1.0 and neg_condition is not None
        for i in range(num_steps):
            # NOT clamped to self.t_max: sigmas[0] is exactly 1.0 and the reference pipeline
            # runs its first step there. (The clamp was numerically invisible anyway --
            # bf16(0.999) == 1.0, since the bf16 grid spacing near 1.0 is 0.0039 -- so it was
            # never the source of trajectory drift; that was the timestep cast chain, see
            # `forward`.) Dropping it is safe because this loop only ever asks for
            # `fwd_pred_type="flow"`, and convert_model_output returns immediately when
            # src == target, without reaching the schedule's is_t_valid assert.
            t = sigmas[i].expand(B).to(self.noise_scheduler.t_precision)
            v = self(x, t, condition=condition, fwd_pred_type="flow")
            if use_cfg:
                v_neg = self(x, t, condition=neg_condition, fwd_pred_type="flow")
                v = v_neg + cfg_scale * (v - v_neg)
            sigma_next = sigmas[i + 1] if i + 1 < num_steps else torch.zeros_like(sigmas[0])
            x = x + v * (sigma_next - sigmas[i])
        self._expect_role = None
        self.set_role(prev_role)
        return x


# Exactly the target set used by the foveated SFT runs (lora/*.sh), so the student and
# critic adapters cover the same modules as the fused teacher LoRA. Note the single-stream
# blocks expose `attn.to_out` as a bare Linear (vs `to_out.0` in the double blocks), hence
# the explicit per-block entries.
_DEFAULT_LORA_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "linear_in", "linear_out", "to_qkv_mlp_proj",
] + [f"single_transformer_blocks.{i}.attn.to_out" for i in range(20)]


def _parse_model_config(spec: str) -> ModelConfig:
    if spec is None:
        return None
    if os.path.exists(spec):
        return ModelConfig(path=spec)
    assert ":" in spec, f"cannot parse model spec {spec!r}"
    i = spec.rfind(":")
    return ModelConfig(model_id=spec[:i], origin_file_pattern=spec[i + 1 :])
