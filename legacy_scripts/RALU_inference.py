import argparse
import os
from pipeline_flux_RALU import FluxPipeline_RALU
from pipeline_flux_foveation import FluxPipeline_Foveation
from diffusers import FluxPipeline
import torch
import numpy as np
import utils
import random
from torchvision.utils import save_image

def main(args):
    # fix randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pipeline == 'ralu':
        pipe = FluxPipeline_RALU.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        ).to(device)
    elif args.pipeline == 'fov':
        pipe = FluxPipeline_Foveation.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        ).to(device)
    else:
        raise ValueError(f"Invalid pipeline: {args.pipeline}")

    # Set RALU parameters
    # Use --use_RALU_default to apply default RALU parameters; otherwise, set N and e manually.
    pipe.set_params(
        use_RALU_default=args.use_RALU_default,
        level=args.level,
        N=args.N,
        e=args.e,
        up_ratio=args.up_ratio,
    )

    # Generate correlated noise
    gaze_pos = (0, 0)
    mask = utils.create_foveation_mask(args.height // 32, args.width // 32, center=gaze_pos, r=0.2).to(device) # [32, 32]
    save_image(mask.float()[None, None, :], f"{args.output_dir}/mask_{gaze_pos[0]}_{gaze_pos[1]}.png")

    num_upsamples = int(mask.float().sum().item()) # foveated regions
    pipe.generate_noise(device, args.height, args.width, num_upsamples)

    # # Optional: disable progress bar
    # pipe.set_progress_bar_config(disable=True)

    torch.cuda.empty_cache()

    prompt = args.prompt
    prompt = "A highly detailed photograph a book with detailed text filling the whole image, cinematic composition, f/8 aperture"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    image = pipe(
        prompt,
        guidance_scale=args.guidance_scale,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        height=args.height,
        width=args.width,
        mask=mask.view(-1).to(device)
    ).images[0]
    
    image.save(f"{output_dir}/img_{gaze_pos[0]}_{gaze_pos[1]}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FluxPipeline_RALU with custom arguments")

    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--guidance_scale', type=float, default=3.5, help='Guidance scale')
    parser.add_argument('--prompt', type=str, default="a cute puppy is playing in the ground", help='Prompt to generate image')
    parser.add_argument('--output_dir', type=str, default='./outputs/test2', help='Directory to save generated images')
    parser.add_argument('--use_RALU_default', action='store_true', help='using RALU default setting (4x, 7x speedup)')
    parser.add_argument('--level', type=int, default=4, choices=[4, 7], help='RALU speedup level (4x or 7x)')
    parser.add_argument('--N', type=int, default=None, nargs='+', help='Number of steps for each stage (e.g., 5 6 7)')
    parser.add_argument('--e', type=float, default=None, nargs='+', help='End timestep for each stage (e.g., 0.3 0.45 1.0)')
    parser.add_argument('--up_ratio', type=float, default=0.3, help='Upsampling ratio')
    parser.add_argument('--seed', type=int, default=10, help='Random seed for generation')
    parser.add_argument('--pipeline', type=str, default='ralu', choices=['ralu', 'fov'], help='Pipeline to use')

    args = parser.parse_args()
    main(args)