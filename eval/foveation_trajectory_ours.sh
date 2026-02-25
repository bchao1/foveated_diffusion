export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True


step=15000
ckpt_folder="/local/brian/foveated_diffusion/models/flux2_foveation/no_tokenAE_rank_32" # new, no tokenAE
#ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/fix_mask_fov"
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment foveation_trajectory_grid \
    --foveation_trajectory_type polygons \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final/foveation_trajectory_ours_polygons \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt_dataset_path /local/brian/Aesthetic-Train-V2/metadata.csv \
    --num_prompts 1 \
    --num_cols 5 \
    --foveation_outline false \
    --soft_foveation_blend true