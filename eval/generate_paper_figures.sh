# foveation trajectory figure
python generate_paper_figures.py foveation_trajectory \
  --base-dir /home/brianchc/foveated_diffusion/results/eccv2026_final \
  --row1-name foveation_trajectory_naive_no_tokenAE \
  --row2-name foveation_trajectory_naive_no_tokenAE \
  --subfolder-idx 0 1 2 \
  --num-rows 3 \
  --output /home/brianchc/foveated_diffusion/paper_figures/foveation_trajectory_grid.png
exit

# baseline figure
python generate_paper_figures.py baselines \
  --base-dir /local/brian/foveated_diffusion/results \
  --baselines full_high_res_eval full_high_res_eval full_high_res_eval \
  --num-rows 5 \
  --output ../paper_figures/baselines.png
exit