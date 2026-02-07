export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
CUDA_VISIBLE_DEVICES=$1 python FLUX2-foveated-inference.py \
    --experiment circular \
    --decode_mode direct \
    --output_dir /home/brianchc/foveated_diffusion/DiffSynth-Studio/results/0207_lora_tokenAE_valid_prompt \
    --tokenAE_folder /home/brianchc/foveated_diffusion/DiffSynth-Studio/examples/flux2/model_training/models/train/FLUX.2-foveation/tokenAE/step-9000 \
    --lora_checkpoint /home/brianchc/foveated_diffusion/DiffSynth-Studio/examples/flux2/model_training/models/train/FLUX.2-foveation/step-9000.safetensors \
    --seed 42 \
    --prompt "Golden retriever on a grassy meadow. Thick sun-kissed fur, clean and glossy, subtle lighter ruff around the neck. Alert amber eyes, black moist nose, relaxed open mouth with pink tongue. Perky ears tilted forward, sturdy athletic stance, tail held in a gentle arc. Warm late-afternoon backdrop, rolling wild grass bending in the breeze, blurred tree line and distant hills under a pastel sky."