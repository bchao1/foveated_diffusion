import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
from ultralytics import YOLOE
import deepgaze_pytorch

from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

from .transformer_wan import WanTransformer3DModel
from .pipeline_wan import WanPipeline 
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from .resizer_wan import load_wan_resizer
from .utils.taehv import TAEHV
from .utils.seg_video_func import prepare_sam2_tiny_video_predictor

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="Wan repo used to load the pipeline weights.")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for video latent noise sampling.")
    parser.add_argument("--height", type=int, default=480, help="Low-res frame height.")
    parser.add_argument("--width", type=int, default=480, help="Low-res frame width.")
    parser.add_argument("--num_frames", type=int, default=49, help="Video output frame count.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Classifier-free guidance scale.")

    parser.add_argument("--hr_ratio", type=float, default=0.15, help="Target share of salient tokens sent to the HR branch.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Total denoising steps.")
    parser.add_argument("--stg1_steps", type=int, default=15, help="LR steps before mixed-resolution mode.")
    parser.add_argument("--shift", type=float, default=3.0, help="Scheduler shift for the denoising timetable.")
    parser.add_argument("--mask_type", type=str, choices=["base", "w_raft", "w_seg"], default="w_raft", help="Plain saliency or RAFT-refined or segmentation-refined.")

    parser.add_argument("--ckpt_dir", type=str, default="./ckpts", help="Chekpoint directory.")
    parser.add_argument("--prompt", type=str, default="giant sunflower slowly turning toward warm golden sunlight in a summer field", help="Text prompt.")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path.")
    parser.add_argument("--fps", type=int, default=8, help="Saved video frame rate.")

    return parser.parse_args()


def main(args):
    torch_dtype = torch.bfloat16

    #---------------Load DiT---------------------
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32).to(args.device)
    transformer = WanTransformer3DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch_dtype
    )
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=args.shift, use_dynamic_shifting=False, base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096,
    )
    pipe = WanPipeline.from_pretrained(
        args.model_id, vae=vae, transformer=transformer, scheduler=scheduler, torch_dtype=torch_dtype,
    )
    # pipe.vae.enable_tiling()

    pipe.to(args.device)

    #---------------Load Other Models---------------------
    centerbias_template = np.load(os.path.join(args.ckpt_dir, "centerbias_mit1003.npy"))
    centerbias = zoom(
        centerbias_template,
        (args.height / centerbias_template.shape[0], args.width / centerbias_template.shape[1]),
        order=0,
        mode="nearest",
    )
    centerbias -= logsumexp(centerbias)
    pipe.centerbias_tensor = torch.from_numpy(centerbias).unsqueeze(0).to(args.device)  # [1, h, w]
    pipe.gaze_model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(args.device)

    if args.mask_type == "w_seg":
        pipe.seg_model = YOLOE(os.path.join(args.ckpt_dir, "yoloe-26s-seg-pf.pt")).to(args.device)
        pipe.sam2_predictor = prepare_sam2_tiny_video_predictor(args.ckpt_dir, device=args.device)
        pipe.warmup_seg(args.num_frames, args.height, args.width, args.device, torch_dtype)

    pipe.raft = raft_small(weights=Raft_Small_Weights.DEFAULT).to(args.device)

    pipe.tvae = TAEHV(os.path.join(args.ckpt_dir, "taew2_1.pth")).to(torch_dtype).to(args.device)

    pipe.lt_downscaler = load_wan_resizer(
        vae=vae,
        mode="down_s",
        checkpoint_path=os.path.join(args.ckpt_dir, "wan_lt_down.pt"),
        num_res_blocks=3,
        dtype=torch_dtype,
        device=args.device,
    )
    pipe.lt_upscaler = load_wan_resizer(
        vae=vae,
        mode="up_s",
        checkpoint_path=os.path.join(args.ckpt_dir, "wan_lt_up.pt"),
        num_res_blocks=3,
        dtype=torch_dtype,
        device=args.device,
    )

    pipe.bi_upscaler = load_wan_resizer(
        vae=vae,
        mode="up_s",
        checkpoint_path=os.path.join(args.ckpt_dir, "wan_bi_up.pt"),
        num_res_blocks=3,
        dtype=torch_dtype,
        device=args.device,
    )

    pipe.gaze_model.eval()
    pipe.raft.eval()
    pipe.tvae.eval()

    #-----------------inference----------------
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    output = pipe(
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator().manual_seed(args.seed),
        hr_ratio=args.hr_ratio,
        stg1_true_steps=args.stg1_steps,
        mask_type=args.mask_type,
    ).frames[0]
    export_to_video(output, args.output, fps=args.fps)


if __name__ == "__main__":
    main(parse_args())
