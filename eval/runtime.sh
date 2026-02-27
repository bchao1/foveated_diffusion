export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

step=15000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/" # new, no tokenAE
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment runtime \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final_v2/runtime \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --mask_shape square \
    --foveation_outline true \
    --soft_foveation_blend true