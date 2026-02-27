export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

step=5000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/"
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

# Number of subsets for distributed full_eval (e.g. number of GPUs)

# Run full_eval for one subset. Usage: run_subset <subset_idx> [num_subsets]
# e.g. run_subset 0; run_subset 1; run_subset 2; run_subset 3
# or: run_subset 0 4 && run_subset 1 4 && run_subset 2 4 && run_subset 3 4
run_subset() {
    num_subsets=8
    local gpu_id="$1"
    local subset_idx="$2"
    CUDA_VISIBLE_DEVICES=$gpu_id python inference.py \
        --experiment ours \
        --decode_mode merge \
        --output_dir /local/brian/foveated_diffusion/results/ours_r_${3}_eval \
        --seed 42 \
        --tokenAE_training false \
        --lora_checkpoint "${lora_checkpoint}" \
        --full_eval \
        --full_eval_mask square \
        --foveation_outline false \
        --soft_foveation_blend true \
        --prompt_dataset_path /local/brian/Aesthetic-Train-V2/metadata_test.csv \
        --num_subsets "${num_subsets}" \
        --subset_idx "${subset_idx}" \
        --mask_radius $3
}

run_subset 0 0 0.25 &
run_subset 1 1 0.25 &
run_subset 2 2 0.25 &
run_subset 3 3 0.25 &
run_subset 4 4 0.25 &
run_subset 5 5 0.25 &
run_subset 6 6 0.25 &
run_subset 7 7 0.25 &
wait
run_subset 0 0 0.75 &
run_subset 1 1 0.75 &
run_subset 2 2 0.75 &
run_subset 3 3 0.75 &
run_subset 4 4 0.75 &
run_subset 5 5 0.75 &
run_subset 6 6 0.75 &
run_subset 7 7 0.75 &
exit


# Normal run
CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment ours \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final_v2/ours_eval \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --full_eval \
    --full_eval_mask square \
    --foveation_outline true \
    --soft_foveation_blend true \
    --prompt_dataset_path /home/brianchc/foveated_diffusion/eval/100_cinematic_prompts.csv \
    --num_prompts 100 \
    --mask_radius 0.5