#!/usr/bin/env bash
# Full foveated DMD2 run.
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

exp_name="${EXP_NAME:-dmd_4step_rank32_2x}"
TEACHER_LORA="${TEACHER_LORA:-/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-10000.safetensors}"

accelerate launch --gpu_ids "$GPU_IDS" --num_processes "$NUM_GPUS" --mixed_precision no \
  distill/train_dmd.py \
  --teacher_lora_path "$TEACHER_LORA" \
  --lr_downsample_factor 2 \
  --student_steps 4 \
  --student_t_mode flux2 \
  --student_sample_type ode \
  --time_dist shifted \
  --critic_update_freq 5 \
  --student_lr 3e-5 --critic_lr 3e-5 \
  --teacher_cfg_scale 4.0 \
  --student_cfg_scale 4.0 \
  --critic_warmup_steps 500 \
  --batch_size 1 --gradient_accumulation_steps 2 \
  --adam_beta1 0.9 \
  --max_train_steps 10000 \
  --validation_steps 500 --teacher_val_steps 50 \
  --save_steps 1000 --log_steps 25 \
  --use_wandb --wandb_project flux2-foveated-dmd --wandb_run_name "$exp_name" \
  --output_path "/local/brian/foveated_diffusion/models/dmd/$exp_name"
