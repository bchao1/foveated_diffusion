export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

step=5000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/"
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

# Normal run
CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment ours \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final_v2/ours_circular_qualitative_v4 \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --full_eval \
    --full_eval_mask circular \
    --foveation_outline true \
    --soft_foveation_blend true \
    --prompt_dataset_path /home/brianchc/foveated_diffusion/eval/prompts/claude_prompts_extended_v2.csv \
    --num_prompts 100 \
    --mask_radius 0.4
exit
CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment ours \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final_v2/teaser \
    --seed 0 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --full_eval_mask square \
    --foveation_outline true \
    --soft_foveation_blend true \
    --prompt "A lion lounges calmly, displaying its powerful physique, with a soft focus highlighting its majestic mane and intense gaze." \
    --mask_radius 0.5
exit
