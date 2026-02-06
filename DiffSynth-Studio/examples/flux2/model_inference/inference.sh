export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
CUDA_VISIBLE_DEVICES=$1 python FLUX2-foveated-inference.py \
    --experiment circular \
    --decode_mode merge \
    --output_dir /home/brianchc/foveated_diffusion/DiffSynth-Studio/results/circular_merge_v5