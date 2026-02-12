export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
ckpt_folder="/local/brian/foveated_diffusion/models/flux2_foveation/lora_with_tokenAE_nn_warmup_rank_64/"
#ckpt_folder="/home/brianchc/foveated_diffusion/DiffSynth-Studio/examples/flux2/model_training/models/train/FLUX.2-foveation"

# instead of setting manually, use tokenAE_config.yaml

naive_upsample_mode="nearest"
naive_downsample_mode="nearest"
upsampler_variance_correction=1.0
downsampler_variance_correction=1.0
upsampler_normalization_type="ada"
downsampler_normalization_type="ada"
upsampler_num_res_blocks=10
downsampler_num_res_blocks=1

lora_path=${ckpt_folder}/step-9000.safetensors
tokenAE_folder=${ckpt_folder}/tokenAE/step-9000

CUDA_VISIBLE_DEVICES=$1 python FLUX2-foveated-inference.py \
    --experiment circular \
    --decode_mode direct \
    --output_dir /home/brianchc/foveated_diffusion/DiffSynth-Studio/results/0211_temb_test \
    --seed 42 \
    --prompt "Golden retriever on a grassy meadow. Thick sun-kissed fur, clean and glossy, subtle lighter ruff around the neck. Alert amber eyes, black moist nose, relaxed open mouth with pink tongue. Perky ears tilted forward, sturdy athletic stance, tail held in a gentle arc. Warm late-afternoon backdrop, rolling wild grass bending in the breeze, blurred tree line and distant hills under a pastel sky." \
    --naive_upsample_mode ${naive_upsample_mode} \
    --naive_downsample_mode ${naive_downsample_mode} \
    --upsampler_variance_correction ${upsampler_variance_correction} \
    --downsampler_variance_correction ${downsampler_variance_correction} \
    --upsampler_normalization_type ${upsampler_normalization_type} \
    --downsampler_normalization_type ${downsampler_normalization_type} \
    --upsampler_num_res_blocks ${upsampler_num_res_blocks} \
    --downsampler_num_res_blocks ${downsampler_num_res_blocks}