export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
CUDA_VISIBLE_DEVICES=$1 python FLUX2-vae-encode-decode.py \
    --input_path  /home/brianchc/foveated_diffusion/DiffSynth-Studio/examples/flux2/model_inference/image_FLUX.2-klein-base-4B.jpg \
    --output_path /home/brianchc/foveated_diffusion/DiffSynth-Studio/results/vae_reconstructed.png