"""Control: the authors' UNMODIFIED RALU code on FLUX.1-dev, its intended base model.

Imports FluxPipeline_RALU from pipeline_flux_RALU.py as-is and runs their published
level-4 setting (use_RALU_default=True -> N=[5,6,7], e=[0.3,0.45,1.0], their hardcoded
shift/Z constants, up_ratio=0.3, Canny edge selection). Measures mean pixel value against
a plain FLUX.1-dev run on the same prompts.

Purpose: distinguish
  (a) RALU is correctly exposed on FLUX.1 -> the method does not transfer to FLUX.2, and
      our FLUX.2 port's darkness is a property of the transfer, not a bug we introduced;
  (b) RALU is also dark on FLUX.1 -> the expectation/measurement is wrong and the FLUX.2
      port is probably fine.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

MODEL = "/home/brianchc/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"   # local NVMe copy; /miele is NFS at 6.3 MB/s


def brightness(img):
    return np.asarray(img.convert("RGB")).astype(float).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_csv", default="../prompts/claude_prompts.csv")
    ap.add_argument("--num_prompts", type=int, default=8)
    ap.add_argument("--level", type=int, default=4, choices=[4, 7])
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--guidance_scale", type=float, default=3.5)  # their default
    ap.add_argument("--seed", type=int, default=10)               # their default
    ap.add_argument("--output_dir", default="../../results/ralu_compare/flux1_control")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prompts = [r["prompt"] for _, r in pd.read_csv(args.prompts_csv).iterrows()][: args.num_prompts]
    print(f"{len(prompts)} prompts")

    # ---- their code, unmodified ----
    from pipeline_flux_RALU_importable import FluxPipeline_RALU  # nbsp-fixed copy, logic identical

    pipe = FluxPipeline_RALU.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_params(use_RALU_default=True, level=args.level, N=None, e=None, up_ratio=0.3)
    pipe.generate_noise("cuda", args.height, args.width)
    pipe.set_progress_bar_config(disable=True)
    print("RALU params:", {k: v for k, v in pipe.params.items() if k in ("N", "e", "shift", "Z", "up_ratio")})

    ralu = []
    for i, p in enumerate(prompts):
        img = pipe(p, guidance_scale=args.guidance_scale, max_sequence_length=256,
                   generator=torch.Generator("cpu").manual_seed(args.seed),
                   height=args.height, width=args.width).images[0]
        img.save(f"{args.output_dir}/ralu_{i:03d}.png")
        ralu.append(brightness(img))
        print(f"  [ralu {i}] {ralu[-1]:.1f}")
    del pipe
    torch.cuda.empty_cache()

    # ---- plain FLUX.1-dev reference, same prompts/seed ----
    from diffusers import FluxPipeline

    ref = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    ref.set_progress_bar_config(disable=True)
    steps = 50
    plain = []
    for i, p in enumerate(prompts):
        img = ref(p, guidance_scale=args.guidance_scale, max_sequence_length=256,
                  num_inference_steps=steps,
                  generator=torch.Generator("cpu").manual_seed(args.seed),
                  height=args.height, width=args.width).images[0]
        img.save(f"{args.output_dir}/plain_{i:03d}.png")
        plain.append(brightness(img))
        print(f"  [plain {i}] {plain[-1]:.1f}")

    r, q = np.array(ralu), np.array(plain)
    print()
    print(f"FLUX.1-dev, RALU level {args.level} (authors' code, unmodified)")
    print(f"  RALU  n={len(r)} mean {r.mean():6.1f} +/- {r.std():5.1f}")
    print(f"  plain n={len(q)} mean {q.mean():6.1f} +/- {q.std():5.1f}")
    print(f"  ratio RALU/plain = {r.mean()/q.mean():.3f}")
    print(f"  (FLUX.2 port for reference: RALU 27.8 vs plain 77.5, ratio 0.359)")


if __name__ == "__main__":
    main()
