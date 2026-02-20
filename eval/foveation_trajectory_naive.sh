export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

# cnn
naive_upsample_mode="bicubic"
naive_downsample_mode="bicubic"
upsampler_normalization_type="gn"
downsampler_normalization_type="none"
upsampler_num_res_blocks=10
downsampler_num_res_blocks=2

# dit 
upsampler_num_dit_blocks=1
downsampler_num_dit_blocks=1
upsampler_dit_use_temb="false"
downsampler_dit_use_temb="false"

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment foveation_trajectory_grid \
    --decode_mode direct \
    --output_dir ../results/eccv2026_final/foveation_trajectory_naive \
    --seed 42 \
    --prediction_type "clean" \
    --tokenAE_arch "dit" \
    --naive_upsample_mode ${naive_upsample_mode} \
    --naive_downsample_mode ${naive_downsample_mode} \
    --upsampler_normalization_type ${upsampler_normalization_type} \
    --downsampler_normalization_type ${downsampler_normalization_type} \
    --upsampler_num_res_blocks ${upsampler_num_res_blocks} \
    --downsampler_num_res_blocks ${downsampler_num_res_blocks} \
    --upsampler_num_dit_blocks ${upsampler_num_dit_blocks} \
    --downsampler_num_dit_blocks ${downsampler_num_dit_blocks} \
    --upsampler_dit_use_temb ${upsampler_dit_use_temb} \
    --downsampler_dit_use_temb ${downsampler_dit_use_temb} \
    --full_eval \
    --prompt_dataset_path /local/brian/Aesthetic-Train-V2/metadata.csv \
    --num_prompts 6 \
    --num_cols 6