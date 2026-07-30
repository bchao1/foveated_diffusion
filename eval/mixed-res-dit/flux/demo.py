import argparse
import os

import deepgaze_pytorch
import numpy as np
import torch
from diffusers import AutoencoderTiny
from scipy.ndimage import zoom
from scipy.special import logsumexp

from .pipeline_flux import FluxPipeline
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from .transformer_flux import FluxTransformer2DModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev", help="FLUX repo used to load the pipeline weights.")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the two noise streams (`seed` and `seed+1`).")
    parser.add_argument("--height", type=int, default=512, help="Low-res image height.")
    parser.add_argument("--width", type=int, default=512, help="Low-res image width.")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Embedded FLUX guidance scale.")
    parser.add_argument("--hr_ratio", type=float, default=0.30, help="Target share of salient tokens refined in the HR branch.")
    parser.add_argument("--speed_up", type=str, choices=["4x", "6x"], default="4x", help="Named 3-stage schedule preset.")

    parser.add_argument("--ckpt_dir", type=str, default="./ckpts", help="Chekpoint directory.")
    parser.add_argument("--prompt", type=str, default="Plush teddy bear, hyper-detailed soft fabric texture, solid light gray background", help="Text prompt.")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output image path.")

    return parser.parse_args()


def main(args):
    torch_dtype = torch.bfloat16

    # -------------------- load main models -----------------------
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_id,
        subfolder="transformer",
        torch_dtype=torch_dtype,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id,
        subfolder="scheduler",
    )
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
    )
    pipe.to(args.device)

    # -------------------- load other models -----------------------
    pipe.tvae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch_dtype).to(args.device)

    gaze_shape = (args.height, args.width)
    centerbias_template = np.load(os.path.join(args.ckpt_dir, "centerbias_mit1003.npy"))
    centerbias = zoom(
        centerbias_template,
        (gaze_shape[0] / centerbias_template.shape[0], gaze_shape[1] / centerbias_template.shape[1]),
        order=0,
        mode="nearest",
    )
    centerbias -= logsumexp(centerbias)
    pipe.centerbias_tensor = torch.from_numpy(centerbias).unsqueeze(0).to(args.device)
    pipe.gaze_shape = gaze_shape
    pipe.gaze_model = deepgaze_pytorch.DeepGazeI(pretrained=True).to(args.device)
    pipe.gaze_model.eval()

    # -------------------- Speedup configs-----------------------
    if args.speed_up == "4x":
        lr_shift = None
        stage_N = [6, 7, 5]
        stage_e = [0.37, 0.6, 1.0]
    elif args.speed_up == "6x":
        lr_shift = 2.0
        stage_N = [2, 3, 5]
        stage_e = [0.2, 0.5, 1.0]

    if lr_shift is not None:
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.model_id, subfolder="scheduler", use_dynamic_shifting=False, shift=lr_shift,
        )
    pipe.set_3stg_params(N=stage_N, e=stage_e)

    # -------------------- inference -----------------------
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        generator2=torch.Generator("cpu").manual_seed(args.seed + 1),
        hr_ratio=args.hr_ratio,
    ).images[0]
    image.save(args.output)


if __name__ == "__main__":
    main(parse_args())
