CUDA_VISIBLE_DEVICES=$1 python flux_foveation_inference.py \
    --pretrained_model_name_or_path /scratch/m000051-pm04/brianchc/FLUX.1-dev/ \
    --output_dir ../results/0121_dog_circular_start_10_mean_var/ \
    --experiment circular \
    --foveation_start_step 10
    #--tokenAE_folder /scratch/m000051-pm04/brianchc/flux_foveation/model_checkpoints/0103_vary_r_pos_valid_foveation/checkpoint-10000