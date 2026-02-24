export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

base_dir="/local/brian/foveated_diffusion/models/flux2_foveation"
model_name="no_tokenAE_rank_32"
step=14000
lora_checkpoint="${base_dir}/${model_name}/step-${step}.safetensors"

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment user_study \
    --prompt_dataset_path /local/brian/Aesthetic-Train-V2/metadata.csv \
    --num_prompts 50 \
    --output_dir ../results/eccv2026_final/user_study \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --foveation_outline false \
    --soft_foveation_blend true \
    --decode_mode merge