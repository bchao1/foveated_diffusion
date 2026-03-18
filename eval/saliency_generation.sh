export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

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

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment foveation_trajectory_grid \
    --foveation_trajectory_type grid \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final_v2/website/fov_traj_grid_10x10/${MODEL} \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt_dataset_path /home/brianchc/foveated_diffusion/eval/prompts/website_prompts_landscapes.csv \
    --num_prompts 100 \
    --orbit_radius 0.3 \
    --mask_radius 0.3 \
    --grid_rows 10 \
    --grid_cols 10 \
    --foveation_outline true \
    --soft_foveation_blend true \
    --outline_width_frac 0.005