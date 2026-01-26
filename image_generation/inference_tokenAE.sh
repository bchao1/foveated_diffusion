CUDA_VISIBLE_DEVICES=$1 python flux_foveation_tokenAE_inference.py \
    --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
    --output_dir ../results/0123_circular_downsample_bilinear_1.0_upsample_bilinear_1.0 \
    --experiment circular
    #--tokenAE_folder /scratch/m000051-pm04/brianchc/flux_foveation/model_checkpoints/0103_vary_r_pos_valid_foveation/checkpoint-10000