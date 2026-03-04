# video frame grid for teaser
for video_id in 24 81 87; do
python generate_paper_figures.py video_grid \
  --folder /home/brianchc/foveated_diffusion/results/video_results/saliency/480p_cam_prompt/${video_id}.mp4 \
  --image-glob "*.png" \
  --num-rows 1 \
  --num-cols 3 \
  --pad-px 20 \
  --output /home/brianchc/foveated_diffusion/paper_figures/videos/baseline/ours_${video_id}.png
done
exit

# video for baseline comparisons 
python generate_paper_figures.py baselines_video \
  --base-dir /home/brianchc/foveated_diffusion/results/video_results/baseline \
  --baselines high_res naive ours \
  --image-ids 3 11 \
  --num-rows 5 \
  --pad-px 10 \
  --no-vertical-padding \
  --draw-circle-overlay \
  --outline-width-frac 0.005 \
  --outline-color "255,0,0" \
  --num-cols 6 \
  --downsample 2 \
  --output /home/brianchc/foveated_diffusion/paper_figures/baseline_video.png
exit

# saliency 
python generate_paper_figures.py stack \
  --images \
    /home/brianchc/foveated_diffusion/paper_figures/vary_radius_bbox_saliency/028.png \
    /home/brianchc/foveated_diffusion/paper_figures/vary_radius_bbox_saliency/033.png \
    /home/brianchc/foveated_diffusion/paper_figures/vary_radius_bbox_saliency/036.png \
    /home/brianchc/foveated_diffusion/paper_figures/vary_radius_bbox_saliency/038.png \
    /home/brianchc/foveated_diffusion/paper_figures/vary_radius_bbox_saliency/041.png \
    /home/brianchc/foveated_diffusion/paper_figures/vary_radius_bbox_saliency/080.png \
    /home/brianchc/foveated_diffusion/paper_figures/vary_radius_bbox_saliency/057.png \
    /home/brianchc/foveated_diffusion/paper_figures/vary_radius_bbox_saliency/070.png \
    /home/brianchc/foveated_diffusion/paper_figures/vary_radius_bbox_saliency/084.png \
  -o /home/brianchc/foveated_diffusion/paper_figures/saliency_vs_bbox_vary_radius.png \
  --num-rows 3 \
  --num-cols 3 \
  --pad-px 20 \
  --downsample 4
exit
# saliency
# idx 000-099
for idx in $(seq -w 000 099); do
python generate_paper_figures.py foveation_trajectory \
  --base-dir /home/brianchc/foveated_diffusion/results/eccv2026_final_v2/vary_radius \
  --row1-name saliency \
  --row2-name bbox \
  --subfolder-idx $idx \
  --num-rows 3 \
  --output /home/brianchc/foveated_diffusion/paper_figures/vary_radius_bbox_saliency/${idx}.png
done

exit

# user study figure
# For parent folder with subfolders 00000, 00001, ..., each containing frames
python generate_paper_figures.py stack_folders \
  --folder /home/brianchc/foveated_diffusion/results/eccv2026_final_v2/user_study \
  --image-glob "*.png" \
  --pad-px 8 \
  -o /home/brianchc/foveated_diffusion/paper_figures/user_study_stacks
exit

# saliency final figure
python generate_paper_figures.py stack \
  --images \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_multi_object_random_model/000.png \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_multi_object/000.png \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_video_games/000.png \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_robotics_ego_raw_v3/000.png \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_self_driving/003.png \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_drones/015.png \
  -o /home/brianchc/foveated_diffusion/paper_figures/saliency_vs_bbox_vary_radius.png \
  --num-rows 3 \
  --num-cols 3 \
  --pad-px 20
exit 


# saliency
# idx 000-099
for idx in $(seq -w 000 019); do
python generate_paper_figures.py foveation_trajectory \
  --base-dir /home/brianchc/foveated_diffusion/results/eccv2026_final_v2 \
  --row1-name saliency_generation_self_driving_v2 \
  --row2-name "" \
  --subfolder-idx $idx \
  --num-rows 3 \
  --output /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_self_driving_v2/${idx}.png
done

exit


# baseline figure
python generate_paper_figures.py baselines \
  --base-dir /home/brianchc/foveated_diffusion/results/eccv2026_final_v2/ \
  --baselines baseline_qualitative naive_circular_qualitative_outlined ours_circular_qualitative_outlined \
  --num-rows 4 \
  --image-ids 8 10 29 30 35 49 61 63 \
  --downsample 4 \
  --output ../paper_figures/baselines.png
exit

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