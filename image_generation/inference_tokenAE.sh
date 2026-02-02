CUDA_VISIBLE_DEVICES=$1 python flux_foveation_tokenAE_inference.py \
    --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
    --output_dir ../results/0202_x0_pred_direct_upsample_nearest \
    --experiment circular
    #--tokenAE_folder /scratch/m000051-pm04/brianchc/flux_foveation/model_checkpoints/0103_vary_r_pos_valid_foveation/checkpoint-10000