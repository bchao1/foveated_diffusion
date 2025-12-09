import argparse
import os
from pipeline_flux_warp import FluxWarpPipeline
import torch
import numpy as np
import utils
import random
from torchvision.utils import save_image
import imageio

def main(args):
    # fix randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # warp mask visualization
    warp_mask = utils.create_warp_mask(args.height // 16, args.width // 16) * 0.1 # flow field
    img = utils.visualize_2d_map(warp_mask)
    img = (img * 255).astype(np.uint8)
    imageio.imwrite(f"warp_mask.png", img)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipe = FluxWarpPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to(device)


    # # Optional: disable progress bar
    # pipe.set_progress_bar_config(disable=True)

    torch.cuda.empty_cache()

    prompt = args.prompt
    prompt = "A highly detailed photograph of a racecar parked on a road, cinematic composition"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    image = pipe(
        prompt,
        guidance_scale=args.guidance_scale,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        height=args.height,
        width=args.width,
        warp_mask=warp_mask
    ).images[0]
    
    image.save(f"{output_dir}/img.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FluxPipeline with custom arguments")

    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--guidance_scale', type=float, default=3.5, help='Guidance scale')
    parser.add_argument('--prompt', type=str, default="a cute puppy is playing in the ground", help='Prompt to generate image')
    parser.add_argument('--output_dir', type=str, default='./outputs/test_flux_warp', help='Directory to save generated images')
    parser.add_argument('--seed', type=int, default=10, help='Random seed for generation')

    args = parser.parse_args()
    main(args)