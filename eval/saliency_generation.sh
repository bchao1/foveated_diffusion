export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

step=5000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_saliency/"
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment foveation_trajectory_grid \
    --foveation_trajectory_type circular \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final_v2/saliency_generation \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt_dataset_path /home/brianchc/foveated_diffusion/eval/prompts/claude_prompts_saliency.csv \
    --prompt_ids 27 33 41 46 \
    --num_cols 20 \
    --foveation_outline true \
    --soft_foveation_blend true \
    --outline_width_frac 0.005