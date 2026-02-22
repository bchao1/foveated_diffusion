# foveation trajectory figure
python generate_paper_figures.py foveation_trajectory \
    --row1-name foveation_trajectory_naive_no_tokenAE \
    --row2-name foveation_trajectory_ours_no_tokenAE \
    --subfolder-idx 0 1 2 3 4 5 6 7 8 9 \
    -o ../paper_figures/trajectory.png \
    --downsample 4