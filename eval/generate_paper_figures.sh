python generate_paper_figures.py baselines_video \
  --base-dir /home/brianchc/foveated_diffusion/results/video_results/baseline \
  --baselines high_res naive ours \
  --image-ids 3 6 \
  --pad-px 10 \
  --no-horizontal-padding \
  --outline-width-frac 0.005 \
  --outline-color "255,0,0" \
  --num-cols 5 \
  --downsample 2 \
  --output /home/brianchc/foveated_diffusion/paper_figures/baseline_video.png
# 3 57 77 84
exit

# video frame grid for teaser
for video_folder in 720p_cam_prompt 720p_no_cam_prompt; do
  for video_id in 53 56 57 62 75 79 81 85 87 88 91; do
    echo "Generating video grid for ${video_folder}/${video_id}"
    python generate_paper_figures.py video_grid \
      --folder /home/brianchc/foveated_diffusion/results/video_results/saliency/${video_folder}/${video_id}.mp4 \
      --image-glob "*.png" \
      --num-rows 1 \
      --num-cols 5 \
      --pad-px 2 \
      --output /home/brianchc/foveated_diffusion/paper_figures/videos/saliency_v2/${video_folder}_${video_id}.png
  done
done
exit


# saliency final figure
python generate_paper_figures.py stack \
  --images \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_random_model/042.png \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation/042.png \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_video_games/000.png \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_robotics_ego_raw_v3/000.png \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_self_driving/003.png \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_drones/015.png \
  -o /home/brianchc/foveated_diffusion/paper_figures/saliency_final.png \
  --num-rows 1 \
  --num-cols 6 \
  --pad-px 20 \
  --downsample 2
# multi-object 000.png
exit 


# saliency vs random
# idx 000-099
for idx in $(seq -w 000 099); do
python generate_paper_figures.py stack \
  --images \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation/${idx}.png \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_random_model/${idx}.png \
  -o /home/brianchc/foveated_diffusion/paper_figures/saliency_vs_random/${idx}.png \
  --num-rows 1 \
  --num-cols 2 \
  --pad-px 20 \
  --downsample 4 \
  --output /home/brianchc/foveated_diffusion/paper_figures/salinecy_vs_random/${idx}.png
done

# all single video for baseline comparisons 
for id in $(seq 1 100); do
python generate_paper_figures.py baselines_video \
  --base-dir /home/brianchc/foveated_diffusion/results/video_results/baseline \
  --baselines high_res naive ours \
  --image-ids $id \
  --pad-px 10 \
  --no-horizontal-padding \
  --outline-width-frac 0.005 \
  --outline-color "255,0,0" \
  --num-cols 6 \
  --downsample 2 \
  --output /home/brianchc/foveated_diffusion/paper_figures/baseline_video/${id}.png
done
exit


# randomized baseline figure
python generate_paper_figures.py stack \
  --images \
    /home/brianchc/foveated_diffusion/results/eccv2026_final_v2/baseline_comparison_randomized/stacked/00008.png \
    /home/brianchc/foveated_diffusion/results/eccv2026_final_v2/baseline_comparison_randomized/stacked/00010.png \
    /home/brianchc/foveated_diffusion/results/eccv2026_final_v2/baseline_comparison_randomized/stacked/00035.png \
    /home/brianchc/foveated_diffusion/results/eccv2026_final_v2/baseline_comparison_randomized/stacked/00029.png \
    /home/brianchc/foveated_diffusion/results/eccv2026_final_v2/baseline_comparison_randomized/stacked/00061.png \
    /home/brianchc/foveated_diffusion/results/eccv2026_final_v2/baseline_comparison_randomized/stacked/00003.png \
  -o /home/brianchc/foveated_diffusion/paper_figures/baseline_comparison_randomized.png \
  --num-rows 3 \
  --num-cols 2 \
  --pad-px 20 \
  --downsample 2
exit


# stacked randomized baseline figure
python generate_paper_figures.py stack_folders \
  --folder /home/brianchc/foveated_diffusion/results/eccv2026_final_v2/baseline_comparison_randomized \
  --image-glob "img*.png" \
  --pad-px 20 \
  -o /home/brianchc/foveated_diffusion/results/eccv2026_final_v2/baseline_comparison_randomized/stacked
exit

# video frame grid for teaser
for video_folder in 720p_cam_prompt 720p_no_cam_prompt; do
  for video_id in 53 56 57 62 75 79 81 85 87 88 91; do
    echo "Generating video grid for ${video_folder}/${video_id}"
    python generate_paper_figures.py video_grid \
      --folder /home/brianchc/foveated_diffusion/results/video_results/saliency/${video_folder}/${video_id}.mp4 \
      --image-glob "*.png" \
      --num-rows 1 \
      --num-cols 5 \
      --pad-px 2 \
      --output /home/brianchc/foveated_diffusion/paper_figures/videos/saliency_v2/${video_folder}_${video_id}.png
  done
done
exit

# video frame grid for teaser
for video_folder in 720p_cam_prompt 720p_no_cam_prompt; do
  for video_id in $(seq 1 100); do
    echo "Generating video grid for ${video_folder}/${video_id}"
    python generate_paper_figures.py video_grid \
      --folder /home/brianchc/foveated_diffusion/results/video_results/saliency/${video_folder}/${video_id}.mp4 \
      --image-glob "*.png" \
      --num-rows 1 \
      --num-cols 3 \
      --pad-px 5 \
      --center-crop-aspect "4:3" \
      --output /home/brianchc/foveated_diffusion/paper_figures/videos/${video_folder}/ours_${video_id}.png
  done
done
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
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation_random_model/084.png \
    /home/brianchc/foveated_diffusion/paper_figures/saliency_generation/084.png \
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