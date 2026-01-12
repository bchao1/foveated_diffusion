python flux_foveation_tokenAE_inference.py \
    --pretrained_model_name_or_path /scratch/m000051-pm04/brianchc/FLUX.1-dev \
    --output_dir ../results/0108_inference_no_train_nn_upsample_x0.5/ \
    --experiment radius_ablation
    #--tokenAE_folder /scratch/m000051-pm04/brianchc/flux_foveation/model_checkpoints/0103_vary_r_pos_valid_foveation/checkpoint-10000