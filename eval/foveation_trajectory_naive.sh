export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

step=15000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/" # new, no tokenAE
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

GPU_ID=${1:-0}

foveation_trajectory_type="multi_circle"
CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
    --experiment foveation_trajectory_grid \
    --foveation_trajectory_type ${foveation_trajectory_type} \
    --decode_mode merge \
    --output_dir ../results/eccv2026_rebuttal/multi_object_1000_prompts/naive \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt_dataset_path /home/brianchc/IQA_metrics/T2I-CompBench/examples/dataset/complex_prompts.csv \
    --num_prompts 1000 \
    --orbit_radius 0.3 \
    --mask_radius 0.3 \
    --num_cols 3 \
    --foveation_outline false \
    --soft_foveation_blend true \
    --outline_width_frac 0.000 \
    --lr_downsample_factor 2