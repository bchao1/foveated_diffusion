export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

step=5000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/"
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"


foveation_trajectory_type="multi_circle"
CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment foveation_trajectory_grid \
    --foveation_trajectory_type ${foveation_trajectory_type} \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final_v2/supp/foveation_trajectory/ours_${foveation_trajectory_type} \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt_dataset_path /home/brianchc/foveated_diffusion/eval/prompts/supp/general_claude_3.csv \
    --num_prompts 100 \
    --num_cols 3 \
    --foveation_outline true \
    --soft_foveation_blend true \
    --outline_width_frac 0.005

# 'radius', 'circular', 'polygons', 'multi_circle