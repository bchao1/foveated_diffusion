#!/usr/bin/env bash
# Standalone launcher for the foveated-diffusion web GUI.
#
# Bakes in the same env vars, conda env, GPU pinning, and checkpoint paths used
# by tests/_common.sh + tests/test_*.sh, so the GUI loads the same model as the
# tests do.
#
# All four LoRAs are registered with the server; the user picks which one is
# active from the dropdown in the page (no CKPT env var needed).
#
# Usage:
#   bash webgui/run.sh                      # GPU 1, port 5000, default LoRA = random
#   GPU_ID=2 bash webgui/run.sh             # pick GPU
#   PORT=8080 bash webgui/run.sh            # pick port
#   DEFAULT_LORA=saliency bash webgui/run.sh  # which LoRA is fused at startup
#   bash webgui/run.sh --port 8080          # extra flags forwarded to server.py

set -e

# --- env matches tests/_common.sh ---
export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"

# --- resolve paths relative to this script ---
WEBGUI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd "$WEBGUI_DIR/.." && pwd)"

# --- checkpoints (same paths as tests/_common.sh) ---
CKPT_NO_FOV="/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/step-15000.safetensors"
CKPT_RANDOM="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-5000.safetensors"
CKPT_SALIENCY="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_saliency/step-11000.safetensors"
CKPT_BBOX="/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_bbox/step-11000.safetensors"

DEFAULT_LORA="${DEFAULT_LORA:-random}"
PORT="${PORT:-5000}"
HOST="${HOST:-0.0.0.0}"

echo "[webgui] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[webgui] default_lora=$DEFAULT_LORA  host=$HOST  port=$PORT"
echo "[webgui] cwd=$RELEASE_DIR"

cd "$RELEASE_DIR"
exec conda run -n imp_tome --no-capture-output python webgui/server.py \
    --host "$HOST" --port "$PORT" \
    --lora "no_fov=$CKPT_NO_FOV" \
    --lora "random=$CKPT_RANDOM" \
    --lora "saliency=$CKPT_SALIENCY" \
    --lora "bbox=$CKPT_BBOX" \
    --default_lora "$DEFAULT_LORA" \
    "$@"
