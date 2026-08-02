"""Foveated DMD / DMD2 objective.

The algorithm is FastGen's `DMD2Model.single_train_step` restructured for a single shared
DiT with three adapter roles, and for a *variable-length* mixed-resolution token sequence.
The two loss functions are imported verbatim from FastGen so the maths is theirs, not a
re-derivation:

    generator step:  L = VSD(gen_x0, teacher_x0, critic_x0)
    critic step:     L = DSM(critic_pred, x0=gen_x0.detach(), eps, t)

Per iteration exactly one of the two runs, alternating on `critic_update_freq`, exactly as
`DMD2Model._setup_grad_requirements` / `get_optimizers` do.

Notes specific to foveation:
  * one foveation mask is drawn per iteration and shared across the batch, because the
    sequence length L depends on the mask and tokens from different masks cannot be stacked.
  * `real_data` is the merged clean foveated latent z0_fov (high-res tokens in the fovea,
    E(Down(x)) tokens in the periphery) -- the same target the SFT loss builds.
  * noise is `randn_like(real_data)` rather than `randn(B, *input_shape)`; a static
    `input_shape` is meaningless when L varies.
"""

from __future__ import annotations

import os
import sys
import math
from typing import Optional

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "DiffSynth-Studio") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "DiffSynth-Studio"))

from fastgen.methods.common_loss import (  # noqa: E402
    denoising_score_matching_loss,
    variational_score_distillation_loss,
)

from .fov_net import Flux2FoveatedNet


class FoveatedDMD(torch.nn.Module):
    """Wraps a `Flux2FoveatedNet` and produces the DMD generator / critic losses."""

    def __init__(
        self,
        net: Flux2FoveatedNet,
        student_steps: int = 4,
        student_t_mode: str = "flux2",
        student_mu: Optional[float] = None,
        student_sample_type: str = "sde",
        critic_update_freq: int = 5,
        critic_warmup_steps: int = 100,
        time_dist: str = "shifted",
        time_shift: float = math.exp(0.8),
        min_t: float = 0.001,
        max_t: float = 0.999,
        teacher_cfg_scale: float = 4.0,
        student_cfg_scale: float = 1.0,
        negative_prompt: str = "",
    ):
        super().__init__()
        self.net = net
        self.student_steps = student_steps
        self.student_sample_type = student_sample_type
        self.critic_update_freq = critic_update_freq
        self.critic_warmup_steps = critic_warmup_steps
        self.time_dist = time_dist
        # FLUX.2's shifted schedule with mu=0.8 IS FastGen's "shifted" distribution with
        # shift=exp(mu): sigma = e^mu*s/(s*(e^mu-1)+1) == t*shift/(t*(shift-1)+1). Sampling t
        # this way matches the density the pretrained model and the SFT LoRA were trained
        # under; uniform-in-t (FastGen's default) puts gradient weight where the teacher saw
        # comparatively few training samples.
        self.time_shift = time_shift
        self.min_t, self.max_t = min_t, max_t
        self.teacher_cfg_scale = teacher_cfg_scale
        # If the student is to be SAMPLED with CFG, then the DMD "generator" is
        # student+CFG, so the generator forward must be CFG-combined and differentiable
        # through both branches. Keeping guidance fixed on both sides isolates step
        # distillation; setting this to 1.0 additionally asks the student to absorb
        # guidance (two entangled objectives).
        self.student_cfg_scale = student_cfg_scale
        self.negative_prompt = negative_prompt
        self.register_buffer(
            "t_list", net.student_t_list(student_steps, student_t_mode, mu=student_mu),
            persistent=False
        )
        self._neg_text = None

    # ------------------------------------------------------------------ utils
    @property
    def sched(self):
        return self.net.noise_scheduler

    def is_generator_iter(self, it: int) -> bool:
        """Update the student on every `critic_update_freq`-th iteration (DMD2 default: the
        critic gets the other iterations so it tracks the moving generator).

        The first `critic_warmup_steps` iterations are critic-only. Without this the very
        first generator step has an *exactly zero* gradient: the critic LoRA is zero-init, so
        critic_x0 == teacher_x0 and the VSD gradient (critic - teacher) vanishes. Warming the
        critic first means the student's first update carries real signal.
        """
        if it < self.critic_warmup_steps:
            return False
        return (it - self.critic_warmup_steps) % self.critic_update_freq == 0

    def sample_t(self, batch_size: int, device) -> torch.Tensor:
        return self.sched.sample_t(
            batch_size,
            time_dist_type=self.time_dist,
            min_t=self.min_t,
            max_t=self.max_t,
            shift=self.time_shift,
            device=device,
        )

    def _student_input(self, real_data: torch.Tensor) -> tuple:
        """Noise / timestep the student is asked to denoise from.

        Single-step: pure noise at t_max. Multi-step: a real sample noised to one of the
        student's own t_list entries (DMD2's multistep trick -- the student must be able to
        start from any of its sampling timesteps).
        """
        B = real_data.shape[0]
        eps = torch.randn_like(real_data)
        if self.student_steps == 1:
            t_s = torch.full(
                (B,), self.sched.max_t, device=real_data.device, dtype=self.sched.t_precision
            )
            return self.sched.latents(noise=eps), t_s
        idx = torch.randint(0, self.student_steps, (B,), device=real_data.device)
        t_s = self.t_list.to(real_data.device)[idx].to(self.sched.t_precision)
        return self.sched.forward_process(real_data, eps, t_s), t_s

    def _student_x0(self, x_t, t, condition, neg_condition=None):
        """Student x0 prediction, CFG-combined when `student_cfg_scale != 1`.

        Gradients flow through BOTH branches -- the generator being distilled is the
        guided sampler, not the bare network.
        """
        self.net.set_role("student")
        x0 = self.net(x_t, t, condition=condition, fwd_pred_type="x0")
        if self.student_cfg_scale != 1.0:
            if neg_condition is None:
                raise ValueError(
                    f"student_cfg_scale={self.student_cfg_scale} but no neg_condition given"
                )
            x0_neg = self.net(x_t, t, condition=neg_condition, fwd_pred_type="x0")
            x0 = x0_neg + self.student_cfg_scale * (x0 - x0_neg)
        return x0

    def _teacher_x0(self, x_t, t, condition, neg_condition=None):
        """The DMD "real score", CFG-guided in x0 space.

        This model has no guidance embedding, so CFG here is not optional decoration -- it is
        the entire source of prompt adherence in the target distribution. An unguided teacher
        would distil an unguided student. The combination below is algebraically identical to
        FastGen's `_apply_classifier_free_guidance`: w*x0 - (w-1)*x0_neg.

        Only the teacher uses CFG; the student and critic run single-forward, since
        internalising the guidance is the point of the distillation.
        """
        if self.teacher_cfg_scale != 1.0 and neg_condition is None:
            raise ValueError(
                f"teacher_cfg_scale={self.teacher_cfg_scale} but no neg_condition was given; "
                "the teacher would be unguided."
            )
        self.net.set_role("teacher")
        with torch.no_grad():
            x0 = self.net(x_t, t, condition=condition, fwd_pred_type="x0")
            if self.teacher_cfg_scale != 1.0:
                x0_neg = self.net(x_t, t, condition=neg_condition, fwd_pred_type="x0")
                x0 = x0_neg + self.teacher_cfg_scale * (x0 - x0_neg)
        return x0.detach()

    # ------------------------------------------------------------------ steps
    # NOTE on role handling: each loss leaves the *trainable* adapter active on return, and
    # the caller must run `backward()` in that state. Activation checkpointing replays the
    # blocks during backward, so the adapter active at backward time has to be the one that
    # built the graph -- otherwise recomputation produces a different tensor count and
    # torch raises CheckpointError. The no-grad forwards in between are safe because
    # `use_gradient_checkpointing and torch.is_grad_enabled()` disables checkpointing there.
    def generator_loss(self, real_data, condition, neg_condition=None) -> tuple[torch.Tensor, dict]:
        B = real_data.shape[0]
        input_student, t_student = self._student_input(real_data)
        t = self.sample_t(B, real_data.device)
        eps = torch.randn_like(real_data)

        gen_x0 = self._student_x0(input_student, t_student, condition, neg_condition)
        perturbed = self.sched.forward_process(gen_x0, eps, t)

        self.net.set_role("critic")
        with torch.no_grad():
            critic_x0 = self.net(perturbed, t, condition=condition, fwd_pred_type="x0")

        teacher_x0 = self._teacher_x0(perturbed, t, condition, neg_condition)
        self.net.set_role("student")  # required before backward (see NOTE above)

        loss = variational_score_distillation_loss(gen_x0, teacher_x0, critic_x0.detach())
        logs = {
            "loss/vsd": loss.detach(),
            "diag/teacher_gen_l1": (teacher_x0 - gen_x0).abs().mean().detach(),
            "diag/critic_teacher_l1": (critic_x0 - teacher_x0).abs().mean().detach(),
            "diag/gen_x0_std": gen_x0.detach().float().std(),
            "diag/real_x0_std": real_data.detach().float().std(),
            "diag/t": t.float().mean(),
            "diag/t_student": t_student.float().mean(),
        }
        return loss, logs

    def critic_loss(self, real_data, condition, neg_condition=None) -> tuple[torch.Tensor, dict]:
        B = real_data.shape[0]
        input_student, t_student = self._student_input(real_data)
        t = self.sample_t(B, real_data.device)
        eps = torch.randn_like(real_data)

        with torch.no_grad():
            gen_x0 = self._student_x0(input_student, t_student, condition, neg_condition)
        x_t = self.sched.forward_process(gen_x0, eps, t)

        self.net.set_role("critic")  # stays active through backward (see NOTE above)
        pred = self.net(x_t, t, condition=condition, fwd_pred_type="x0")
        loss = denoising_score_matching_loss(
            "x0", net_pred=pred, noise_scheduler=self.sched, x0=gen_x0, eps=eps, t=t
        )
        return loss, {"loss/critic": loss.detach()}
