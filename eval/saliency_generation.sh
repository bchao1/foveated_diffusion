export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

# saliency model
step=9000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_saliency/"
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

# random model
#step=5000
#ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/"
#lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment foveation_trajectory_grid \
    --foveation_trajectory_type circular \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final_v2/saliency_generation_drones \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt_dataset_path /home/brianchc/foveated_diffusion/eval/prompts/drones.csv \
    --num_prompts 20 \
    --num_cols 5 \
    --foveation_outline true \
    --soft_foveation_blend true \
    --outline_width_frac 0.005