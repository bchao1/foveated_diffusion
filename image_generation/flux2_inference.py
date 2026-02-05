import argparse
import os
import sys
sys.path.append("..")
from pipeline_flux2_klein import Flux2KleinPipeline
from diffusers.models import Flux2Transformer2DModel
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

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    ).to(device)

    pipe = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to(device)


    # # Optional: disable progress bar
    # pipe.set_progress_bar_config(disable=True)

    torch.cuda.empty_cache()

    prompt = args.prompt
    prompt = "A highly detailed photograph of a racecar parked on a road, cinematic composition"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    image = pipe(
        prompt=prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        height=args.height,
        width=args.width,
        max_sequence_length=256,
    ).images[0]
    
    image.save(f"{output_dir}/img.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flux2KleinPipeline with custom arguments")

    parser.add_argument('--pretrained_model_name_or_path', type=str, default='black-forest-labs/FLUX.2-klein-base-4B', help='Path or HuggingFace model ID')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--guidance_scale', type=float, default=4.0, help='Guidance scale (FLUX2 default is 4.0)')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of denoising steps')
    parser.add_argument('--prompt', type=str, default="a cute puppy is playing in the ground", help='Prompt to generate image')
    parser.add_argument('--output_dir', type=str, default='./outputs/test_flux2_klein', help='Directory to save generated images')
    parser.add_argument('--seed', type=int, default=10, help='Random seed for generation')

    args = parser.parse_args()
    main(args)
