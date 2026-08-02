"""Verify `Flux2FoveatedNet.sample_teacher` against `Flux2FoveatedImagePipeline.__call__`.

This is the strongest available correctness check on the distillation wrapper: if a
from-scratch Euler loop through `Flux2FoveatedNet.forward` reproduces the reference
pipeline pixel-for-pixel from the same seed, then the time convention (t == sigma ==
timestep/1000), the mixed-resolution RoPE ids, the embedded-guidance value, the reduced
token layout and the merge-decode path are all wired correctly.

    python distill/verify_teacher.py --teacher_lora_path ... --num_steps 50
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill.fov_net import Flux2FoveatedNet, circular_mask  # noqa: E402
from distill.train_dmd import DEFAULT_MODEL_PATHS, DEFAULT_PROMPT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_lora_path", default=None)
    p.add_argument("--lr_downsample_factor", type=int, default=2)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedded_guidance", type=float, default=4.0)
    p.add_argument("--cfg_scale", type=float, default=1.0,
                   help="True CFG; exercises the negative-prompt branch.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--out_dir", default="/local/brian/foveated_diffusion/models/dmd_smoke/verify")
    args = p.parse_args()

    net = Flux2FoveatedNet(
        model_id_with_origin_paths=DEFAULT_MODEL_PATHS,
        tokenizer_path="black-forest-labs/FLUX.2-klein-base-4B:tokenizer/",
        lora_path=args.teacher_lora_path,
        device="cuda",
        height=args.height,
        width=args.width,
        lr_downsample_factor=args.lr_downsample_factor,
        embedded_guidance=args.embedded_guidance,
        use_gradient_checkpointing=False,
    )
    net.to("cuda")
    net.set_role("teacher")
    pipe = net.pipe
    f = args.lr_downsample_factor
    h, w = net.latent_height, net.latent_width
    mask = circular_mask(h, w, (-0.3, 0.0), 0.25, net.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- reference: the DiffSynth pipeline itself -------------------------------
    ref = pipe(
        prompt=args.prompt,
        negative_prompt="",
        cfg_scale=args.cfg_scale,
        embedded_guidance=args.embedded_guidance,
        height=args.height,
        width=args.width,
        seed=args.seed,
        rand_device="cpu",
        num_inference_steps=args.num_steps,
        foveation_mask=mask,
        decode_mode="merge",
        tokenAE_training=False,
        lr_downsample_factor=f,
    )
    ref.save(os.path.join(args.out_dir, f"reference_pipeline_cfg{args.cfg_scale:g}.png"))

    # ---- ours: rebuild the pipeline's exact initial noise, then our Euler loop ----
    noise = pipe.generate_noise((1, 128, h, w), seed=args.seed, rand_device="cpu",
                                rand_torch_dtype=pipe.torch_dtype)
    noise_d = pipe.generate_noise((1, 128, h // f, w // f), seed=args.seed, rand_device="cpu",
                                  rand_torch_dtype=pipe.torch_dtype)
    noise = noise.reshape(1, 128, h * w).permute(0, 2, 1)
    noise_d = noise_d.reshape(1, 128, (h // f) * (w // f)).permute(0, 2, 1)
    merged = pipe._downsample_latents(
        noise, h, w, foveation_mask=mask, token_downsampler=None,
        latents_downsampled=noise_d, lr_factor=f,
    )
    print(f"[verify] merged noise {tuple(merged.shape)}  (full-res would be {h*w})")

    text = net.encode_prompt([args.prompt])
    cond = net.make_condition(text, mask)
    neg = net.make_condition(net.encode_prompt([""]), mask) if args.cfg_scale != 1.0 else None
    z = net.sample_teacher(merged, cond, num_steps=args.num_steps,
                           cfg_scale=args.cfg_scale, neg_condition=neg)
    ours = net.decode_fov_latents(z, mask)[0]
    ours.save(os.path.join(args.out_dir, f"ours_sample_teacher_cfg{args.cfg_scale:g}.png"))

    a = np.asarray(ref, dtype=np.float32)
    b = np.asarray(ours, dtype=np.float32)
    mae = np.abs(a - b).mean()
    pmax = np.abs(a - b).max()
    print(f"[verify] pixel MAE = {mae:.4f} / 255   max abs diff = {pmax:.1f} / 255")
    if mae < 1.0:
        print("[verify] PASS -- our sampler reproduces the reference pipeline")
    else:
        print("[verify] FAIL -- sampler diverges from the reference pipeline")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
