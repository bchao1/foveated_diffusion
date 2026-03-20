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
    lr_downsample_factor=2
    ;;
  random_4x)
    step=28000
    ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_4x/"
    lr_downsample_factor=4
    ;;
  random_4x_distill)
    step=30000
    ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_4x_distill/"
    lr_downsample_factor=4    
    ;;
  bbox)
    step=11000
    ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_bbox/"
    lr_downsample_factor=2
    ;;
  saliency)
    step=11000
    ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_saliency/"
    lr_downsample_factor=2
    ;;
  *)
    echo "Unknown model: $MODEL (use: random | bbox | saliency)"
    exit 1
    ;;
esac
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment foveation_trajectory_grid \
    --foveation_trajectory_type grid \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final_v2/website/teaser_grid_10x10_${lr_downsample_factor}x_v2/${MODEL} \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt_dataset_path /home/brianchc/foveated_diffusion/eval/prompts/website_prompts_landscapes_v2.csv \
    --num_prompts 100 \
    --orbit_radius 0.3 \
    --mask_radius 0.3 \
    --grid_rows 10 \
    --grid_cols 10 \
    --foveation_outline false \
    --soft_foveation_blend true \
    --outline_width_frac 0.005 \
    --lr_downsample_factor ${lr_downsample_factor}