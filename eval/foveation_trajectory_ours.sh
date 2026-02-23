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


step=10000
#ckpt_folder="/local/brian/foveated_diffusion/models/flux2_foveation/adapter_arch_bilinear/" # cnn
ckpt_folder="/local/brian/foveated_diffusion/models/flux2_foveation/dit_tokenAE_1_block_1_head_no_temb/" # dit
ckpt_folder="/local/brian/foveated_diffusion/models/flux2_foveation/no_tokenAE_test" # new, no tokenAE
tokenAE_folder="${ckpt_folder}/tokenAE/step-${step}"
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

for rank in 32 64 128; do
ckpt_folder="/local/brian/foveated_diffusion/models/flux2_foveation/no_tokenAE_rank_${rank}" # new, no tokenAE
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"
CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment foveation_trajectory_grid \
    --foveation_trajectory_type circular \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final/foveation_trajectory_ours_step_${step}_no_tokenAE_rank_${rank} \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt_dataset_path /local/brian/Aesthetic-Train-V2/metadata.csv \
    --num_prompts 10 \
    --num_cols 5 \
    --foveation_outline false \
    --soft_foveation_blend false
done