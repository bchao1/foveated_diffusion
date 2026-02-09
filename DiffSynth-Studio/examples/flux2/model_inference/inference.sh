export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
ckpt_folder="/local/brian/foveated_diffusion/models/flux2_foveation/lora_no_tokenAE"
#ckpt_folder="/home/brianchc/foveated_diffusion/DiffSynth-Studio/examples/flux2/model_training/models/train/FLUX.2-foveation"

CUDA_VISIBLE_DEVICES=$1 python FLUX2-foveated-inference.py \
    --experiment circular \
    --decode_mode merge \
    --output_dir /home/brianchc/foveated_diffusion/DiffSynth-Studio/results/0209_no_tokenAE_inference_lora_only_circular_merge \
    --seed 42 \
    --prompt "Golden retriever on a grassy meadow. Thick sun-kissed fur, clean and glossy, subtle lighter ruff around the neck. Alert amber eyes, black moist nose, relaxed open mouth with pink tongue. Perky ears tilted forward, sturdy athletic stance, tail held in a gentle arc. Warm late-afternoon backdrop, rolling wild grass bending in the breeze, blurred tree line and distant hills under a pastel sky." \
    --lora_checkpoint $ckpt_folder/step-9000.safetensors