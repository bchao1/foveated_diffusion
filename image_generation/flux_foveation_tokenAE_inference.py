import argparse
import os
import sys
sys.path.append("..")
from pipeline_flux_foveation_tokenAE import FluxPipeline_Foveation
from transformer_flux_crpa import FluxPhaseAlignedTransformer2DModel
import torch
import numpy as np
import utils
import random
from torchvision.utils import save_image
from utils import create_foveation_mask
import torch.nn.functional as F
from adapter_foveation import TokenDownsampler, TokenUpsampler

def main(args):
    # fix randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transformer = FluxPhaseAlignedTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    ).to(device)

    pipe = FluxPipeline_Foveation.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to(device)

    token_upsampler = TokenUpsampler(in_channels=16, out_channels=16, num_res_blocks=10, naive_upsample_mode="bilinear")
    token_downsampler = TokenDownsampler(in_channels=64, out_channels=64, num_res_blocks=10)
    if args.tokenAE_folder is not None:
        token_upsampler = torch.load(os.path.join(args.tokenAE_folder, "token_upsampler.pt"), map_location='cpu', weights_only=False)
        token_downsampler = torch.load(os.path.join(args.tokenAE_folder, "token_downsampler.pt"), map_location='cpu', weights_only=False)
    token_upsampler = token_upsampler.to(device).bfloat16()
    token_downsampler = token_downsampler.to(device).bfloat16()
    

    prompt = args.prompt
    prompt = "A highly detailed photograph of a racecar parked on a road, cinematic composition"


    if args.experiment == "position_ablation":
        for y_idx, center_y in enumerate(np.linspace(-0.25, 0.25, 5)):
            for x_idx, center_x in enumerate(np.linspace(-0.25, 0.25, 5)):
                center = (center_x, center_y)
                foveation_mask = create_foveation_mask(args.height // 16 // 2, args.width // 16 // 2, center=center, r = 0.3) # low-res foveation mask then upsample
                foveation_mask = foveation_mask.unsqueeze(0).unsqueeze(0).float() # [1, 1, H, W]
                save_image(foveation_mask, f"{output_dir}/foveation_mask_low_res_{x_idx}_{y_idx}.png")
                foveation_mask = F.interpolate(foveation_mask, size=(args.height // 16, args.width // 16), mode='nearest')
                save_image(foveation_mask, f"{output_dir}/foveation_mask_high_res_{x_idx}_{y_idx}.png")


                # # Optional: disable progress bar
                # pipe.set_progress_bar_config(disable=True)

                torch.cuda.empty_cache()

                image = pipe(
                    prompt,
                    guidance_scale=args.guidance_scale,
                    max_sequence_length=256,
                    generator=torch.Generator("cpu").manual_seed(args.seed),
                    height=args.height,
                    width=args.width,
                    foveation_mask=foveation_mask.squeeze(), # [H, W]
                    foveation_start_step=args.foveation_start_step
                    #foveation_mask=None
                ).images[0]
                
                image.save(f"{output_dir}/img_{x_idx}_{y_idx}.png")
    elif args.experiment == "radius_ablation":
        r_list = np.linspace(0.1, 1.0, 10)
        r_list = [r_list[1]]
        for r_idx, r in enumerate(r_list):
            r_idx = str(r_idx).zfill(2)

            # low-res foveation mask
            foveation_mask = create_foveation_mask(args.height // 16 // 2, args.width // 16 // 2, center=(0, 0), r = r) # low-res foveation mask then upsample
            foveation_mask_low_res = foveation_mask.clone()

            # create high-res foveation mask
            foveation_mask = foveation_mask.unsqueeze(0).unsqueeze(0).float() # [1, 1, H, W]

            #save_image(foveation_mask, f"{output_dir}/foveation_mask_low_res_{r_idx}.png")
            foveation_mask_high_res = F.interpolate(foveation_mask, size=(args.height // 16, args.width // 16), mode='nearest')
            save_image(foveation_mask, f"{output_dir}/foveation_mask_high_res_{r_idx}.png")
            foveation_mask_high_res = foveation_mask_high_res.squeeze()

            image = pipe(
                prompt,
                guidance_scale=args.guidance_scale,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(args.seed),
                height=args.height,
                width=args.width,
                foveation_mask=foveation_mask_high_res, # [H, W]
                foveation_start_step=args.foveation_start_step,
                token_upsampler=token_upsampler,
                token_downsampler=token_downsampler
                #foveation_mask=None
            ).images[0]
            image.save(f"{output_dir}/img_{r_idx}.png")
    elif args.experiment == "baseline":
        image = pipe(
            prompt,
            guidance_scale=args.guidance_scale,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            height=args.height,
            width=args.width,
            foveation_mask=None
        ).images[0]
        image.save(f"{output_dir}/img.png")

    
    utils.make_videos(output_dir, f"{output_dir}/combined.mp4", fps=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FluxPipeline with custom arguments")

    parser.add_argument('--pretrained_model_name_or_path', required=True)
    parser.add_argument('--tokenAE_folder', default=None, type=str)
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--guidance_scale', type=float, default=3.5, help='Guidance scale')
    parser.add_argument('--prompt', type=str, default="a cute puppy is playing in the ground", help='Prompt to generate image')
    parser.add_argument('--output_dir', type=str, default='./outputs/test_flux_base', help='Directory to save generated images')
    parser.add_argument('--seed', type=int, default=10, help='Random seed for generation')
    parser.add_argument('--experiment', type=str, default="position_ablation", help='Experiment to run')
    parser.add_argument('--foveation_start_step', type=int, default=0, help='Step to start foveation')

    args = parser.parse_args()
    main(args)