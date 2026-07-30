import argparse
import os

import numpy as np
import torch

import deepgaze_pytorch
from scipy.ndimage import zoom
from scipy.special import logsumexp
from ultralytics import YOLOE

from diffusers import AutoencoderTiny

from .pipeline_flux_2stg import FluxPipeline
from .resizer_flux import instantiate_resizer
from .transformer_flux_2t import FluxTransformer2DModel
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev", help="FLUX repo used to load the pipeline weights.")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for latent noise sampling.")
    parser.add_argument("--height", type=int, default=512, help="Low-res image height.")
    parser.add_argument("--width", type=int, default=512, help="Low-res image width.")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Embedded FLUX guidance scale.")

    parser.add_argument("--stg1_steps", type=int, default=6, help="LR-only denoising steps before mask construction.")
    parser.add_argument("--stg2_steps", type=int, default=9, help="Mixed-resolution denoising steps after stage 1.")
    parser.add_argument("--lr_shift", type=float, default=1.8776105792643432, help="Scheduler shift for the LR timetable.") # same as original FLUX
    parser.add_argument("--hr_shift", type=float, default=3.0, help="Scheduler shift for the HR timetable.")
    parser.add_argument("--hr_ratio", type=float, default=0.6, help="Target share of salient tokens sent to the HR branch.")
    parser.add_argument("--mask_type", type=str, choices=["base", "w_seg"], default="w_seg", help="Plain saliency or segmentation-refined.")

    parser.add_argument("--ckpt_dir", type=str, default="./ckpts", help="Chekpoint directory.")
    parser.add_argument("--prompt", type=str, default="a black cat", help="Text prompt.")
    parser.add_argument("--output", type=str, default="output_2stg.jpg", help="Output image path.")

    return parser.parse_args()

def main(args):
    torch_dtype = torch.bfloat16

    # -------------------- load main models -----------------------
    transformer = FluxTransformer2DModel.from_pretrained(args.model_id, subfolder="transformer", torch_dtype=torch_dtype)
    pipe = FluxPipeline.from_pretrained(args.model_id, transformer=transformer, torch_dtype=torch_dtype)
    pipe.to(args.device)

    # -------------------- load other models -----------------------
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler", use_dynamic_shifting=False, shift=args.lr_shift
    )
    pipe.scheduler_hr = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler", use_dynamic_shifting=False, shift=args.hr_shift
    )

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

    if args.mask_type == "w_seg":
        pipe.seg_model = YOLOE(os.path.join(args.ckpt_dir, "yoloe-26s-seg-pf.pt")).to(args.device)

    pipe.lt_upscaler = instantiate_resizer(
        ckpt_path=os.path.join(args.ckpt_dir, "flux_lt_up.safetensors"),
        resizer_type="up",
        norm_type="layer_norm",
        device=args.device,
        dtype=torch_dtype,
    )
    pipe.lt_downscaler = instantiate_resizer(
        ckpt_path=os.path.join(args.ckpt_dir, "flux_lt_down.safetensors"),
        resizer_type="down",
        norm_type="layer_norm",
        device=args.device,
        dtype=torch_dtype,
    )
    pipe.bi_upscaler = instantiate_resizer(
        ckpt_path=os.path.join(args.ckpt_dir, "flux_bi_up.safetensors"),
        resizer_type="up",
        norm_type="layer_norm",
        device=args.device,
        dtype=torch_dtype,
    )

    # -------------------- inference -----------------------
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        hr_ratio=args.hr_ratio,
        stg1_true_steps=args.stg1_steps,
        stg2_true_steps=args.stg2_steps,
        mask_type=args.mask_type,
    ).images[0]
    image.save(args.output)


if __name__ == "__main__":
    main(parse_args())
