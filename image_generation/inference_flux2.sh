CUDA_VISIBLE_DEVICES=$1 python flux2_foveation_tokenAE_inference.py \
    --pretrained_model_name_or_path /miele/brian/modelscope/black-forest-labs/FLUX.2-klein-base-4B/ \
    --output_dir ../results/0204_flux2_klein_base_4B_foveation_merge \
    --experiment circular \
    --decode_mode merge
    #--tokenAE_folder /scratch/m000051-pm04/brianchc/flux_foveation/model_checkpoints/0103_vary_r_pos_valid_foveation/checkpoint-10000