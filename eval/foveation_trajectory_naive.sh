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
    --foveation_trajectory_type circular \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final/foveation_trajectory_naive_no_tokenAE \
    --seed 42 \
    --tokenAE_training false \
    --prompt_dataset_path /local/brian/Aesthetic-Train-V2/metadata.csv \
    --num_prompts 10 \
    --num_cols 5