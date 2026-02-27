# Baseline inference: paths aligned with base.sh (root_folder for data/output/ckpt)
root_folder="/scratch/m000051-pm05/brianchc_2"

step=15000
ckpt_folder="${root_folder}/foveated_diffusion/experiments/finetuned_no_fov/"
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

run_subset() {
    num_subsets=8
    local gpu_id="$1"
    local subset_idx="$2"
    CUDA_VISIBLE_DEVICES=$gpu_id python /users/brianchc/foveated_diffusion/eval/inference.py \
        --experiment high_res \
        --decode_mode direct \
        --tokenAE_training false \
        --output_dir "${root_folder}/foveated_diffusion/results/full_high_res_eval" \
        --seed 42 \
        --lora_checkpoint "${lora_checkpoint}" \
        --full_eval \
        --prompt_dataset_path "${root_folder}/Aesthetic-Train-V2/metadata_test.csv" \
        --num_subsets "${num_subsets}" \
        --subset_idx "${subset_idx}"
}

run_subset 0 0 &
run_subset 1 1 &
run_subset 2 2 &
run_subset 3 3 &
run_subset 4 4 &
run_subset 5 5 &
run_subset 6 6 &
run_subset 7 7 &

wait
