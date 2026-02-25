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
        --experiment naive_mixed_res \
        --decode_mode merge \
        --tokenAE_training false \
        --output_dir /local/brian/foveated_diffusion/results/naive_mixed_res_circle_eval \
        --seed 42 \
        --lora_checkpoint "${lora_checkpoint}" \
        --full_eval \
        --full_eval_mask circular \
        --foveation_outline false \
        --soft_foveation_blend true \
        --prompt_dataset_path /local/brian/Aesthetic-Train-V2/metadata_test.csv \
        --num_subsets "${num_subsets}" \
        --subset_idx "${subset_idx}" \
        --mask_radius 0.5
}


CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment naive_mixed_res \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final_v2/naive_eval \
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