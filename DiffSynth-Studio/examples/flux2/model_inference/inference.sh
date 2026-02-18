export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True
ckpt_folder="/local/brian/foveated_diffusion/models/flux2_foveation/adapter_arch_bilinear"
#ckpt_folder="/home/brianchc/foveated_diffusion/DiffSynth-Studio/examples/flux2/model_training/models/train/FLUX.2-foveation"

# instead of setting manually, use tokenAE_config.yaml

naive_upsample_mode="bilinear"
naive_downsample_mode="bilinear"
upsampler_normalization_type="gn"
downsampler_normalization_type="none"
upsampler_num_res_blocks=10
downsampler_num_res_blocks=2

step=15000
lora_checkpoint=${ckpt_folder}/step-${step}.safetensors
tokenAE_folder=${ckpt_folder}/tokenAE/step-${step}

CUDA_VISIBLE_DEVICES=$1 python FLUX2-foveated-inference.py \
    --experiment circular \
    --decode_mode direct \
    --output_dir /home/brianchc/foveated_diffusion/DiffSynth-Studio/paper_figures/clean_pred_direct_bilinear \
    --seed 42 \
    --prompt "Golden retriever on a grassy meadow. Thick sun-kissed fur, clean and glossy, subtle lighter ruff around the neck. Alert amber eyes, black moist nose, relaxed open mouth with pink tongue. Perky ears tilted forward, sturdy athletic stance, tail held in a gentle arc. Warm late-afternoon backdrop, rolling wild grass bending in the breeze, blurred tree line and distant hills under a pastel sky." \
    --prediction_type "clean" \
    --tokenAE_arch "adapter" \
    --naive_upsample_mode ${naive_upsample_mode} \
    --naive_downsample_mode ${naive_downsample_mode} \
    --upsampler_normalization_type ${upsampler_normalization_type} \
    --downsampler_normalization_type ${downsampler_normalization_type} \
    --upsampler_num_res_blocks ${upsampler_num_res_blocks} \
    --downsampler_num_res_blocks ${downsampler_num_res_blocks} \
    --save_vis_vlow_x0 \
    --vis_vlow_x0_steps "0,12,25,37,49"
    #--tokenAE_arch "adapter" \
    #--tokenAE_folder $tokenAE_folder \
    #--lora_checkpoint $lora_checkpoint
    #--naive_upsample_mode ${naive_upsample_mode} \
    #--naive_downsample_mode ${naive_downsample_mode} \
    #--upsampler_variance_correction ${upsampler_variance_correction} \
    #--downsampler_variance_correction ${downsampler_variance_correction} \
    #--upsampler_normalization_type ${upsampler_normalization_type} \
    #--downsampler_normalization_type ${downsampler_normalization_type} \
    #--upsampler_num_res_blocks ${upsampler_num_res_blocks} \
    #--downsampler_num_res_blocks ${downsampler_num_res_blocks}