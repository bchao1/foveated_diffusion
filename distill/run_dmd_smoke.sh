#!/usr/bin/env bash
# 4-GPU smoke test: correctness assertions + a few alternating DMD steps + one validation.
set -euo pipefail
export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTHONPATH="/home/brianchc/FastGen:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
# Shared machine: pin to genuinely idle GPUs. Check occupancy FIRST with
#   nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
# NOTE: must be passed as `accelerate launch --gpu_ids`, because accelerate overwrites
# CUDA_VISIBLE_DEVICES itself (default --gpu_ids=all -> rank0 lands on physical GPU 0).
GPU_IDS="${GPU_IDS:-3,4,5,6}"
NUM_GPUS="${NUM_GPUS:-$(awk -F, '{print NF}' <<< "$GPU_IDS")}"
cd "$(dirname "$0")/.."

TEACHER_LORA="${TEACHER_LORA:-/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-10000.safetensors}"

accelerate launch --gpu_ids "$GPU_IDS" --num_processes "$NUM_GPUS" --mixed_precision no \
  distill/train_dmd.py \
  --smoke_test \
  --teacher_lora_path "$TEACHER_LORA" \
  --lr_downsample_factor 2 \
  --student_steps 4 \
  --critic_update_freq 5 \
  --batch_size 1 \
  --max_samples 64 \
  --max_train_steps 10 \
  --log_steps 1 \
  --validation_steps 0 \
  --save_steps 0 \
  --teacher_val_steps 8 \
  --output_path /local/brian/foveated_diffusion/models/dmd_smoke
