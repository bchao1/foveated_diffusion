python extract_inception_features.py \
    --output_dir /home/howard/foveated_diffusion/results/metric_foveated_center/ \
    --images_dir /local/brian/Aesthetic-Train-V2/ \
    --batch_size 512 \
    --device cuda:8,cuda:9 \
    --foveate_real \
    --masks_dir /local/brian/Aesthetic-Train-V2/