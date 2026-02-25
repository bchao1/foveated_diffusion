export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

step=15000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/" # new, no tokenAE
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

run_subset() {
    num_subsets=8
    local gpu_id="$1"
    local subset_idx="$2"
    CUDA_VISIBLE_DEVICES=$gpu_id python inference.py \
        --experiment high_res \
        --decode_mode direct \
        --tokenAE_training false \
        --output_dir /local/brian/foveated_diffusion/results/full_high_res_eval \
        --seed 42 \
        --lora_checkpoint "${lora_checkpoint}" \
        --full_eval \
        --prompt_dataset_path /local/brian/Aesthetic-Train-V2/metadata_test.csv \
        --num_subsets "${num_subsets}" \
        --subset_idx "${subset_idx}"
}

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment high_res \
    --decode_mode direct \
    --tokenAE_training false \
    --output_dir ../results/eccv2026_final_v2/baseline_eval \
    --seed 42 \
    --lora_checkpoint "${lora_checkpoint}" \
    --full_eval \
    --prompt_dataset_path /home/brianchc/foveated_diffusion/eval/100_cinematic_prompts.csv \
    --num_prompts 100