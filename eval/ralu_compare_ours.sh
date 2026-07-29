#!/bin/bash
# Single-prompt reference runs for the RALU comparison.
# Matches the RALU-foveated setting: circular mask, r=0.3, lr_downsample_factor=2.
export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True
# imp_tome resolves `diffsynth` to /home/brianchc/DiffSynth-Studio, which lacks the
# foveated pipeline. Force the private repo's checkout.
export PYTHONPATH=/home/brianchc/foveated_diffusion-private/DiffSynth-Studio

PROMPT="A lion lounges calmly, displaying its powerful physique, with a soft focus highlighting its majestic mane and intense gaze."
OUT=../results/ralu_compare
PY=/home/brianchc/miniconda3/envs/imp_tome/bin/python

fov_lora="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-5000.safetensors"
nofov_lora="/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/step-15000.safetensors"

CUDA_VISIBLE_DEVICES=0 $PY inference.py \
    --experiment ours \
    --decode_mode merge \
    --output_dir ${OUT}/ours \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${fov_lora} \
    --full_eval_mask circular \
    --foveation_outline false \
    --soft_foveation_blend true \
    --prompt "${PROMPT}" \
    --mask_radius 0.3 \
    --lr_downsample_factor 2 &

CUDA_VISIBLE_DEVICES=1 $PY inference.py \
    --experiment naive_mixed_res \
    --decode_mode merge \
    --output_dir ${OUT}/naive \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${nofov_lora} \
    --full_eval_mask circular \
    --foveation_outline false \
    --soft_foveation_blend true \
    --prompt "${PROMPT}" \
    --mask_radius 0.3 \
    --lr_downsample_factor 2 &

CUDA_VISIBLE_DEVICES=2 $PY inference.py \
    --experiment high_res \
    --decode_mode direct \
    --output_dir ${OUT}/high_res \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${nofov_lora} \
    --prompt "${PROMPT}" &

wait
