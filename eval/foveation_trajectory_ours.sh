export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

step=5000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/"
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"


foveation_trajectory_type="grid"
CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment foveation_trajectory_grid \
    --foveation_trajectory_type ${foveation_trajectory_type} \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final_v2/supp/foveation_trajectory/ours_${foveation_trajectory_type} \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt "A lion lounges calmly, displaying its powerful physique, with a soft focus highlighting its majestic mane and intense gaze." \
    --orbit_radius 0.3 \
    --mask_radius 0.3 \
    --grid_rows 5 \
    --grid_cols 5 \
    --foveation_outline true \
    --soft_foveation_blend true \
    --outline_width_frac 0.005 

# 'radius', 'circular', 'polygons', 'multi_circle