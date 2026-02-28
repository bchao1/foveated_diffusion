export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

# Usage: ./vary_radius.sh <gpu_id> <model>
#   model: random | bbox | saliency  (selects checkpoint and output subdir)
GPU_ID=${1:-0}
MODEL=${2:?Usage: $0 <gpu_id> <model>   model must be: random | bbox | saliency}

case "$MODEL" in
  random)
    step=5000
    ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/"
    ;;
  bbox)
    step=11000
    ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_bbox/"
    ;;
  saliency)
    step=11000
    ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_saliency/"
    ;;
  *)
    echo "Unknown model: $MODEL (use: random | bbox | saliency)"
    exit 1
    ;;
esac
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
    --experiment foveation_trajectory_grid \
    --foveation_trajectory_type radius \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final_v2/vary_radius/${MODEL} \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt_dataset_path /home/brianchc/foveated_diffusion/eval/prompts/saliency.csv \
    --num_prompts 100 \
    --num_cols 5 \
    --foveation_outline true \
    --soft_foveation_blend true \
    --outline_width_frac 0.005