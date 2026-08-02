"""At initialisation the student IS the teacher (zero-init LoRA), so any difference in the
4-step image is purely the sampler. This quantifies that.

Claim under test: the `ode` re-noising branch is algebraically the Euler update, so
student(ode) at init should match teacher(Euler) on the same timestep grid, whereas the
`sde` branch (fresh noise each step) is much worse for an *undistilled* model because every
x0 prediction is the blurry conditional mean.

    python distill/verify_student_sampler.py --teacher_lora_path ...
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill.fov_net import Flux2FoveatedNet, circular_mask, flux2_sigmas  # noqa: E402
from distill.train_dmd import DEFAULT_MODEL_PATHS, DEFAULT_PROMPT  # noqa: E402


def mae(a, b):
    return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_lora_path", default=None)
    p.add_argument("--lr_downsample_factor", type=int, default=2)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="/local/brian/foveated_diffusion/models/dmd_smoke/sampler")
    args = p.parse_args()

    net = Flux2FoveatedNet(
        model_id_with_origin_paths=DEFAULT_MODEL_PATHS,
        tokenizer_path="black-forest-labs/FLUX.2-klein-base-4B:tokenizer/",
        lora_path=args.teacher_lora_path,
        device="cuda",
        lr_downsample_factor=args.lr_downsample_factor,
        use_gradient_checkpointing=False,
    )
    net.to("cuda")
    net.set_role("teacher")  # == the student at init (zero-init LoRA)
    os.makedirs(args.out_dir, exist_ok=True)

    h, w = net.latent_height, net.latent_width
    mask = circular_mask(h, w, (0.0, 0.0), 0.25, net.device)
    text = net.encode_prompt([DEFAULT_PROMPT])
    negt = net.encode_prompt([""])
    cond = net.make_condition(text, mask)
    neg = net.make_condition(negt, mask)
    L = net.token_count(mask)

    def noise():
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        return torch.randn(1, L, 128, generator=g).to(net.device, net.torch_dtype)

    def gen():
        return torch.Generator(device="cpu").manual_seed(args.seed + 1)

    # reference: 50-step Euler teacher with CFG
    ref = net.decode_fov_latents(
        net.sample_teacher(noise(), cond, 50, cfg_scale=args.cfg_scale, neg_condition=neg), mask
    )[0]
    ref.save(f"{args.out_dir}/teacher_50.png")

    out = {}
    # teacher few-step Euler on its native shifted grid (the current baseline)
    z = net.sample_teacher(noise(), cond, args.steps, cfg_scale=args.cfg_scale, neg_condition=neg)
    out[f"teacher {args.steps}-step Euler (flux2 grid)"] = net.decode_fov_latents(z, mask)[0]

    # student sampler, both branches, on both grids -- no CFG (student is single-forward)
    for mode in ("uniform", "flux2"):
        t_list = net.student_t_list(args.steps, mode)
        for stype in ("ode", "sde"):
            z = net.sample_student(noise(), cond, t_list, sample_type=stype, generator=gen())
            out[f"student {args.steps}-step {stype} ({mode} grid)"] = net.decode_fov_latents(z, mask)[0]

    # teacher Euler with NO cfg, for an apples-to-apples control against the student
    z = net.sample_teacher(noise(), cond, args.steps, cfg_scale=1.0)
    out[f"teacher {args.steps}-step Euler (flux2 grid, no cfg)"] = net.decode_fov_latents(z, mask)[0]

    print(f"\nMAE vs teacher-50 reference (L={L}, cfg={args.cfg_scale} where noted):")
    for k, img in out.items():
        img.save(f"{args.out_dir}/{k.replace(' ', '_').replace('(', '').replace(')', '')}.png")
        print(f"  {mae(img, ref):7.3f}   {k}")

    # the direct claim: ode student == Euler teacher on the SAME grid, no cfg either side
    t_list = net.student_t_list(args.steps, "flux2")
    z_ode = net.sample_student(noise(), cond, t_list, sample_type="ode")
    img_ode = net.decode_fov_latents(z_ode, mask)[0]
    img_eul = out[f"teacher {args.steps}-step Euler (flux2 grid, no cfg)"]
    print(f"\n[claim] student(ode, flux2 grid) vs teacher(Euler, flux2 grid), both no-cfg: "
          f"MAE = {mae(img_ode, img_eul):.3f} / 255")
    print("        (should be ~0 if the ode branch is the Euler update)")


if __name__ == "__main__":
    main()
