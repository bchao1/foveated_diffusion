export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment high_res \
    --output_dir ../results/eccv2026_final/baseline_eval \
    --seed 42 \
    --full_eval \
    --prompt_dataset_path /local/brian/Aesthetic-Train-V2/metadata.csv \
    --num_prompts 20