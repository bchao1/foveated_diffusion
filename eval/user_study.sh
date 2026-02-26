export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

step=5000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/"
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment user_study \
    --prompt_dataset_path /home/brianchc/foveated_diffusion/eval/prompts/claude_prompts_saliency.csv \
    --num_prompts 100 \
    --output_dir ../results/eccv2026_final/user_study_v3 \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --foveation_outline false \
    --soft_foveation_blend true \
    --decode_mode merge