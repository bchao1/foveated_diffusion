python generate_metrics.py \
    --generated_images_dir /local/brian/foveated_diffusion/results/full_high_res_eval/ \
    --images_dataset_path /local/brian/Aesthetic-Train-V2/ \
    --device cuda:8,cuda:9 \
    --metric_output_dir /home/howard/foveated_diffusion/results/metric_high_res/ \
    --batch_size 512 \