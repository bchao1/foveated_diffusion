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
  random_4x)
    step=23000
    ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_4x/"
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
    --output_dir ../results/eccv2026_final_v2/website/teaser_radius_1x/${MODEL} \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt "A lion lounges calmly, displaying its powerful physique, with a soft focus highlighting its majestic mane and intense gaze." \
    --num_cols 1 \
    --foveation_outline false \
    --soft_foveation_blend true \
    --outline_width_frac 0.005 \
    --lr_downsample_factor 1