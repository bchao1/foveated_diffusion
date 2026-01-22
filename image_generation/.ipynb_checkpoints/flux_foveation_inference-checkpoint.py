import argparse
import os
import sys
sys.path.append("..")
from pipeline_flux_foveation import FluxPipeline_Foveation
from transformer_flux_crpa import FluxPhaseAlignedTransformer2DModel
import torch
import numpy as np
import utils
import random
from torchvision.utils import save_image
from utils import create_foveation_mask
import torch.nn.functional as F
import time
from matplotlib import pyplot as plt
import math

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

    prompt = args.prompt
    prompt = "A highly detailed photograph of a racecar parked on a road, cinematic composition"
    prompt = "Documentary-style imagery: a lively little dog stands still on a lush green lawn, filling the entire frame. The dog has brownish-yellow fur, with both ears perked up, and an expression that is focused and cheerful. Sunlight evenly illuminates its body, making the fur appear soft, glossy, and highly detailed. Every part of the scene is in sharp focus, from the dog’s fur to the surrounding grass and distant background elements. The background shows an open grassy field with scattered wildflowers, and faint blue sky with a few white clouds, all rendered clearly and crisply. Strong realism and clarity, emphasizing texture, color, and detail throughout the image. Full-frame composition, static pose, no depth-of-field blur, everything sharp."


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

                start_time = time.time()
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
                end_time = time.time()
                print(f"Time taken: {end_time - start_time} seconds")
                
                image.save(f"{output_dir}/img_{x_idx}_{y_idx}.png")
    elif args.experiment == "circular":        
        # --- Configuration -----------------------------------------
        num_steps = 100          # Number of frames in one full rotation
        orbit_radius = 0.25     # DISTANCE: How far the mask is from the center (0.0 = center, 0.5 = edge)
        mask_radius = 0.30      # SIZE: The radius of the foveation bubble itself
        # -----------------------------------------------------------
        
        print(f"Generating {num_steps} frames. Orbit Radius: {orbit_radius}, Mask Radius: {mask_radius}")
        
        # Iterate through 360 degrees (0 to 2pi radians)
        for i in range(num_steps):
            
            # 1. Calculate the current Angle
            angle = 2 * math.pi * (i / num_steps)
            
            # 2. Compute Center Position (Polar -> Cartesian)
            # This determines WHERE the mask is located
            center_x = orbit_radius * math.cos(angle)
            center_y = orbit_radius * math.sin(angle)
            center = (center_x, center_y)
        
            # 3. Create the Mask
            # Note: We pass 'mask_radius' to 'r' here to set the SIZE of the circle
            foveation_mask = create_foveation_mask(
                args.height // 16 // 2, 
                args.width // 16 // 2, 
                center=center, 
                r=mask_radius  # <--- This sets the size of the mask bubble
            )
            
            # 4. Process dimensions [1, 1, H, W]
            foveation_mask = foveation_mask.unsqueeze(0).unsqueeze(0).float() 
            
            # Save Debug Visualizations
            save_image(foveation_mask, f"{output_dir}/foveation_mask_low_res_{i:03d}.png")
            
            # Upsample for the pipeline
            foveation_mask_high = F.interpolate(
                foveation_mask, 
                size=(args.height // 16, args.width // 16), 
                mode='nearest'
            )
            save_image(foveation_mask_high, f"{output_dir}/foveation_mask_high_res_{i:03d}.png")
        
            # 5. Run Diffusion Pipeline
            torch.cuda.empty_cache()
            start_time = time.time()
            
            image = pipe(
                prompt,
                guidance_scale=args.guidance_scale,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(args.seed),
                height=args.height,
                width=args.width,
                foveation_mask=foveation_mask_high.squeeze(), 
                foveation_start_step=args.foveation_start_step
            ).images[0]
            
            end_time = time.time()
            print(f"Step {i:02d}/{num_steps} | Angle: {math.degrees(angle):.1f}° | Time: {end_time - start_time:.2f}s")
            
            # Save final output
            image.save(f"{output_dir}/img_{i:03d}.png")
    elif args.experiment == "radius_ablation":
        r_list = np.linspace(0.1, 1.0, 100)
        time_list = []
        #r_list = [0.3]
        percentage_HR_list = []
        for r_idx, r in enumerate(r_list):
            r_idx = str(r_idx).zfill(2)
            foveation_mask = create_foveation_mask(args.height // 16 // 2, args.width // 16 // 2, center=(0, 0), r = r) # low-res foveation mask then upsample
            foveation_mask = foveation_mask.unsqueeze(0).unsqueeze(0).float() # [1, 1, H, W]
            save_image(foveation_mask, f"{output_dir}/foveation_mask_low_res_{r_idx}.png")
            foveation_mask = F.interpolate(foveation_mask, size=(args.height // 16, args.width // 16), mode='nearest')
            save_image(foveation_mask, f"{output_dir}/foveation_mask_high_res_{r_idx}.png")

            # percentage of high-res tokens
            percentage_HR = int(foveation_mask.sum() / foveation_mask.numel() * 100)
            print(f"Percentage of high-res tokens: {percentage_HR}")
            percentage_HR_list.append(percentage_HR)
            start_time = time.time()
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
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
            time_list.append(end_time - start_time)
            image.save(f"{output_dir}/img_{r_idx}.png")

        # weird first pipeline call always slower --> sth wrong with kernel issues?
        time_list = time_list[1:-1]
        percentage_HR_list = percentage_HR_list[1:-1]
        print(f"Time list: {time_list}")
        print(f"Percentage HR list: {percentage_HR_list}")
        utils.timing_plot(time_list, percentage_HR_list, output_dir)
        
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

    
    utils.make_videos(output_dir, f"{output_dir}/combined.mp4", fps=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FluxPipeline with custom arguments")

    parser.add_argument('--pretrained_model_name_or_path', required=True)
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