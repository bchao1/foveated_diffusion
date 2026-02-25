# stack foveation trajectory figures
python generate_paper_figures.py stack \
  --images \
    /home/brianchc/foveated_diffusion/paper_figures/foveation_trajectory/circular_1.png \
    /home/brianchc/foveated_diffusion/paper_figures/foveation_trajectory/radius_4.png \
    /home/brianchc/foveated_diffusion/paper_figures/foveation_trajectory/multi_circle_7.png \
  -o /home/brianchc/foveated_diffusion/paper_figures/foveation_trajectory/merged_2.png \
  --axis horizontal \
  --pad-px 20
exit
# foveation trajectory figure
for fov_trajectory_type in radius circular polygons multi_circle; do
  for idx in 0 1 2 3 4 5 6 7 8; do
  python generate_paper_figures.py foveation_trajectory \
    --base-dir /home/brianchc/foveated_diffusion/results/eccv2026_final_v2/foveation_trajectory \
    --row1-name naive_${fov_trajectory_type} \
    --row2-name ours_${fov_trajectory_type} \
    --subfolder-idx $idx \
    --num-rows 3 \
    --output /home/brianchc/foveated_diffusion/paper_figures/foveation_trajectory/${fov_trajectory_type}_${idx}.png
  done
done
exit

# baseline figure
python generate_paper_figures.py baselines \
  --base-dir /home/brianchc/foveated_diffusion/results/eccv2026_final_v2/ \
  --baselines baseline_qualitative_outlined naive_qualitative_outlined ours_qualitative_outlined \
  --num-rows 4 \
  --image-ids 8 10 29 30 35 49 61 63 \
  --output ../paper_figures/baselines.png
exit