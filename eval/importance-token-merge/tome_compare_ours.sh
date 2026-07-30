#!/bin/bash
# Single-prompt reference runs for the ToMe-2x2 comparison.
# Matches the foveated setting used elsewhere: circular mask, r=0.3, lr_downsample_factor=2.
# Mirrors eval/ralu_compare_ours.sh so the two baselines are run under identical conditions.
export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True
# imp_tome resolves `diffsynth` to /home/brianchc/DiffSynth-Studio, a different checkout.
# Force this repo's.
export PYTHONPATH=/home/brianchc/foveated_diffusion-private/DiffSynth-Studio

PROMPT="A lion lounges calmly, displaying its powerful physique, with a soft focus highlighting its majestic mane and intense gaze."
OUT=../../results/tome_compare
PY=/home/brianchc/miniconda3/envs/imp_tome/bin/python

fov_lora="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-5000.safetensors"
nofov_lora="/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/step-15000.safetensors"

# --- ToMe-2x2 baseline, attention-only merging in the double blocks ---
CUDA_VISIBLE_DEVICES=0 $PY tome_fov_inference.py \
    --output_dir ${OUT}/tome_attn \
    --seed 42 \
    --lora_checkpoint ${nofov_lora} \
    --full_eval_mask circular \
    --mask_radius 0.3 \
    --lr_downsample_factor 2 \
    --merge_mlp false \
    --prompt "${PROMPT}" &

# --- ToMe-2x2 baseline, merging the image MLP in the double blocks too ---
CUDA_VISIBLE_DEVICES=1 $PY tome_fov_inference.py \
    --output_dir ${OUT}/tome_attn_mlp \
    --seed 42 \
    --lora_checkpoint ${nofov_lora} \
    --full_eval_mask circular \
    --mask_radius 0.3 \
    --lr_downsample_factor 2 \
    --merge_mlp true \
    --prompt "${PROMPT}" &

# --- Identity control: same code path, merging disabled. Should match high_res. ---
CUDA_VISIBLE_DEVICES=2 $PY tome_fov_inference.py \
    --output_dir ${OUT}/tome_off \
    --seed 42 \
    --lora_checkpoint ${nofov_lora} \
    --enable_tome false \
    --prompt "${PROMPT}" &

# --- Ours, same mask ---
CUDA_VISIBLE_DEVICES=3 $PY ../inference.py \
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

wait
