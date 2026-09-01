# Plan for remaking qualitative figures.

## Figures to remake
- Main qualitative baseline:
    - /home/brianchc/foveated_diffusion-private/paper_figures/baseline_comparison_randomized.png
- Supp Figs: 
    - /home/brianchc/foveated_diffusion-private/paper_figures/supp/baseline_1_5col.png
    - /home/brianchc/foveated_diffusion-private/paper_figures/supp/baseline_2_5col.png
    - /home/brianchc/foveated_diffusion-private/paper_figures/supp/baseline_3_5col.png
    - /home/brianchc/foveated_diffusion-private/paper_figures/supp/baseline_101_5col.png

## Instructions
- Since the TOME and RALU setup might have changed, we need to use the most up-to-date inference script: /home/brianchc/foveated_diffusion-private/eval/eval.sh to re-generate the TOME and RALU images and remake the figures. You can inspect if anything has changed or if not, we can just use the existing figure. You should not re-generate the naive, high res and ours results, and just find the existing images.
- The inference setting and how to set args should just exactly follow the eval.sh script (inference steps, mask radius, shape, etc).
- You would need to inspect the codebase should only be in eval/, paper_figures/, and /results
- You would need to inspect the codebase to find what exact prompt set, indices, etc are used to generated the images.

## Style of the images
- Use serif font labels, roughly one eight image height (singular image, not full image)
- For the main qual figure, it should be six colums, and the ordering is full high-res, *the tokenization mask*, tome, ralu, naive, and ours. Show the horse, dog, and man image, so 3 rows.
    - For the tokenization mask, refer to the color palette defined in /home/brianchc/LoT-lior-branch/inference/flux/generate.py
- For the supplemental figures
    - For baseline 101, there are 4 prompts in total, and the layout is 4x3 grid into one image. Change it into:
        - 2 figures, each with 2 prompts
        - For 1 prompt, the layout is 2x3 in the form of |high res|tome|ralu| (next row) |empty white block|naive|ours|
    - For baseline 1, 2, 3, make it 5 column, and split prompt sets into two sets and get two images, resulting in 6 images
        - If total odd prompts, drop 1 so that can be evenly split into two images.

## Output
- Write a clean single script for qualiattive figures inference and image making and save it to eval/iclr_2027 for reproducibility
- One script for main qual baseline, one for baseline 101, one for baseline 1,2,3, 3 scripts in total

## Env
- Use imp_tome conda env
- Pick empty gpus to use, or the emptiest