CUDA_VISIBLE_DEVICES=$1 python flux_foveation_inference.py \
    --output_dir ../results/0114_fix_packing_training_free_start_step_10_pack_fix/ \
    --experiment radius_ablation \
    --foveation_start_step 10
    #--tokenAE_folder /scratch/m000051-pm04/brianchc/flux_foveation/model_checkpoints/0103_vary_r_pos_valid_foveation/checkpoint-10000