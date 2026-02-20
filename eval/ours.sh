export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

# cnn
naive_upsample_mode="bilinear"
naive_downsample_mode="bilinear"
upsampler_normalization_type="gn"
downsampler_normalization_type="none"
upsampler_num_res_blocks=10
downsampler_num_res_blocks=2

# dit 
upsampler_num_dit_blocks=1
downsampler_num_dit_blocks=1
upsampler_dit_use_temb="false"
downsampler_dit_use_temb="false"


step=7000
ckpt_folder="/local/brian/foveated_diffusion/models/flux2_foveation/adapter_arch_bilinear/" # cnn
ckpt_folder="/local/brian/foveated_diffusion/models/flux2_foveation/dit_tokenAE_1_block_1_head_no_temb/" # dit
tokenAE_folder="${ckpt_folder}/tokenAE/step-${step}"
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment ours \
    --decode_mode merge \
    --output_dir ../results/eccv2026_final/ours_dit_merge_eval \
    --seed 42 \
    --prediction_type "clean" \
    --tokenAE_arch "dit" \
    --tokenAE_folder ${tokenAE_folder} \
    --lora_checkpoint ${lora_checkpoint} \
    --full_eval \
    --prompt_dataset_path /local/brian/Aesthetic-Train-V2/metadata.csv \
    --num_prompts 20