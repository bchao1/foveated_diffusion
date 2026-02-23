export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

step=10000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov/" # new, no tokenAE
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

# Number of subsets for distributed full_eval (e.g. number of GPUs)

# Run full_eval for one subset. Usage: run_subset <subset_idx> [num_subsets]
# e.g. run_subset 0; run_subset 1; run_subset 2; run_subset 3
# or: run_subset 0 4 && run_subset 1 4 && run_subset 2 4 && run_subset 3 4
run_subset() {
    num_subsets=4
    local gpu_id="$1"
    local subset_idx="$2"
    CUDA_VISIBLE_DEVICES=$gpu_id python inference.py \
        --experiment ours \
        --decode_mode merge \
        --output_dir ../results/eccv2026_final/ours_eval \
        --seed 42 \
        --tokenAE_training false \
        --lora_checkpoint "${lora_checkpoint}" \
        --full_eval \
        --full_eval_mask square \
        --foveation_outline false \
        --soft_foveation_blend true \
        --prompt_dataset_path /local/brian/Aesthetic-Train-V2/metadata.csv \
        --num_prompts 50 \
        --num_subsets "${num_subsets}" \
        --subset_idx "${subset_idx}"
}

# Uncomment to run a single subset:
run_subset 0 0 &
run_subset 1 1 &
run_subset 2 2 &
run_subset 3 3 &

exit

# Normal run
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py \
    --experiment ours \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final/ours_no_tokenAE_parallel_test \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --full_eval \
    --full_eval_mask square \
    --foveation_outline false \
    --soft_foveation_blend true \
    --prompt_dataset_path /local/brian/Aesthetic-Train-V2/metadata.csv \
    --num_prompts 50