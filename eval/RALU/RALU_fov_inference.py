"""Driver for the 2-stage RALU-foveated baseline on FLUX.2-klein.

Mask geometry is replicated bit-for-bit from eval/inference.py:create_foveation_mask so
RALU and ours foveate exactly the same region. Guidance is off by default: klein-base has
no guidance_embedder weights, so our DiffSynth runs are effectively unguided and RALU must
match (see --guidance_scale).
"""
import argparse
import json
import math
import os
import re

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from pipeline_flux2_klein_RALU_fov import Flux2KleinRALUFovPipeline

MODEL_PATH = "/miele/brian/modelscope/black-forest-labs/FLUX.2-klein-base-4B"


def coarse_foveation_mask(height, width, center=(0.0, 0.0), r=0.3, shape="circular", lr_factor=2):
    """The low-res grid that eval/inference.py:create_foveation_mask builds before its
    nearest upsample to the token grid. Identical geometry, no upsample."""
    high_h, high_w = height // 16, width // 16
    low_h, low_w = high_h // lr_factor, high_w // lr_factor

    y_range = torch.arange(low_h, dtype=torch.float32)
    x_range = torch.arange(low_w, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")
    cx = math.floor((center[0] + 0.5) * low_w - 0.5) + 0.5
    cy = math.floor((center[1] + 0.5) * low_h - 0.5) + 0.5

    if shape == "square":
        side = r * min(low_h, low_w)
        half = side / 2.0
        mask = ((x_grid >= cx - half) & (x_grid <= cx + half)
                & (y_grid >= cy - half) & (y_grid <= cy + half))
    elif shape == "circular":
        diagonal = (low_h ** 2 + low_w ** 2) ** 0.5
        radius_px = r * (diagonal / 2.0)
        mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= radius_px ** 2
    else:
        raise ValueError(f"unsupported mask shape for RALU baseline: {shape}")
    return mask.float()


def convert_diffsynth_lora(path):
    """DiffSynth PEFT-style keys -> diffusers. Module names already match; only the
    `transformer.` prefix and the `.default` PEFT adapter infix differ."""
    sd = load_file(path)
    out = {}
    for k, v in sd.items():
        nk = k.replace(".lora_A.default.weight", ".lora_A.weight")
        nk = nk.replace(".lora_B.default.weight", ".lora_B.weight")
        if not nk.startswith("transformer."):
            nk = "transformer." + nk
        out[nk] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guidance_scale", type=float, default=1.0,
                    help="1.0 = no CFG, matching our unguided DiffSynth runs")
    ap.add_argument("--lora_checkpoint", type=str, default=None)
    ap.add_argument("--mask_shape", type=str, default="circular")
    ap.add_argument("--mask_radius", type=float, default=0.3)
    ap.add_argument("--lr_downsample_factor", type=int, default=2)
    ap.add_argument("--N", type=int, nargs="+", default=[25, 25])
    ap.add_argument("--e", type=float, nargs="+", default=[0.35, 1.0])
    ap.add_argument("--id_frame", type=str, default="coarse", choices=["coarse", "full"],
                    help="coarse = RALU-faithful stage-2 frame; full = apply their stage-3 rescale early")
    ap.add_argument("--full_res_check", action="store_true",
                    help="set the mask to all-ones as a sanity check")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    pipe = Flux2KleinRALUFovPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    pipe.vae.enable_tiling()   # their RALU_inference calls the pipeline-level equivalent

    if args.lora_checkpoint:
        sd = convert_diffsynth_lora(args.lora_checkpoint)
        pipe.load_lora_weights(sd, adapter_name="default")
        print(f"loaded LoRA: {args.lora_checkpoint} ({len(sd)} tensors)")

    mask = coarse_foveation_mask(
        args.height, args.width, center=(0, 0), r=args.mask_radius,
        shape=args.mask_shape, lr_factor=args.lr_downsample_factor,
    )
    if args.full_res_check:
        mask = torch.ones_like(mask)
    pipe.id_frame = args.id_frame
    pipe.set_foveation_mask(mask)

    seq_len = (args.height // 16) * (args.width // 16)
    params = pipe.set_params(N=args.N, e=args.e, seq_len=seq_len, f0=args.lr_downsample_factor)
    print(json.dumps({k: v for k, v in params.items()}, indent=2, default=str))

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    pipe.generate_noise(device, args.height, args.width,
                        dtype=torch.bfloat16, generator=gen)

    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        generator=gen,
    ).images[0]

    tag = ("full_res_check" if args.full_res_check else "ralu_fov") + f"_{args.id_frame}"
    image.save(os.path.join(args.output_dir, f"img_{tag}.png"))

    m = mask.numpy()
    n_hr = int(m.sum())
    m_down = m.size
    tokens_stage1 = m_down
    tokens_stage2 = 4 * n_hr + (m_down - n_hr)
    stats = {
        "prompt": args.prompt,
        "params": {k: (v if not isinstance(v, list) else list(map(float, v)))
                   for k, v in params.items()},
        "hr_blocks": n_hr,
        "coarse_blocks": m_down,
        "fovea_area_frac": n_hr / m_down,
        "tokens_stage1": tokens_stage1,
        "tokens_stage2": tokens_stage2,
        "tokens_full_res": (args.height // 16) * (args.width // 16),
        "denoise_time_s": pipe.denoise_time,
        "total_time_s": pipe.total_time,
    }
    with open(os.path.join(args.output_dir, f"stats_{tag}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    F.interpolate(mask[None, None], size=(args.height, args.width), mode="nearest")
    from PIL import Image
    import numpy as np
    Image.fromarray((F.interpolate(mask[None, None], size=(args.height, args.width),
                                   mode="nearest")[0, 0].numpy() * 255).astype(np.uint8)
                    ).save(os.path.join(args.output_dir, "mask.png"))


if __name__ == "__main__":
    main()
