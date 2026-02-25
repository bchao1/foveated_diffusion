export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

step=15000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/" # new, no tokenAE
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment naive_mixed_res \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final_v2/naive_circular_qualitative \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --full_eval \
    --full_eval_mask circular \
    --foveation_outline true \
    --soft_foveation_blend true \
    --prompt_dataset_path /home/brianchc/foveated_diffusion/eval/prompts/claude_prompts.csv \
    --num_prompts 100 \
    --mask_radius 0.4