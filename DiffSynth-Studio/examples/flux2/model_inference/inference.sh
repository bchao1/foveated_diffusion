export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True
ckpt_folder="/local/brian/foveated_diffusion/models/flux2_foveation/adapter_arch_bilinear"

# instead of setting manually, use tokenAE_config.yaml

naive_upsample_mode="bicubic"
naive_downsample_mode="bicubic"
upsampler_normalization_type="gn"
downsampler_normalization_type="none"
upsampler_num_res_blocks=10
downsampler_num_res_blocks=2

# dit 
upsampler_num_dit_blocks=1
downsampler_num_dit_blocks=1
upsampler_dit_use_temb="true"
downsampler_dit_use_temb="true"

step=15000
lora_checkpoint=${ckpt_folder}/step-${step}.safetensors
tokenAE_folder=${ckpt_folder}/tokenAE/step-${step}

CUDA_VISIBLE_DEVICES=$1 python FLUX2-foveated-inference.py \
    --experiment circular \
    --decode_mode direct \
    --output_dir /home/brianchc/foveated_diffusion/DiffSynth-Studio/results/test \
    --seed 42 \
    --prompt "Golden retriever on a grassy meadow. Thick sun-kissed fur, clean and glossy, subtle lighter ruff around the neck. Alert amber eyes, black moist nose, relaxed open mouth with pink tongue. Perky ears tilted forward, sturdy athletic stance, tail held in a gentle arc. Warm late-afternoon backdrop, rolling wild grass bending in the breeze, blurred tree line and distant hills under a pastel sky." \
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
    --save_vis_vlow_x0 \
    --vis_vlow_x0_steps "0,12,25,37,49"