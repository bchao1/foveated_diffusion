export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

step=15000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/" # new, no tokenAE
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment high_res \
    --decode_mode direct \
    --output_dir ../results/eccv2026_final_v2/teaser \
    --seed 30 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt "A lion lounges calmly, displaying its powerful physique, with a soft focus highlighting its majestic mane and intense gaze."
exit 

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment high_res \
    --decode_mode direct \
    --output_dir ../results/eccv2026_final_v2/teaser \
    --seed 0 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt "A lion lounges calmly, displaying its powerful physique, with a soft focus highlighting its majestic mane and intense gaze."
exit
