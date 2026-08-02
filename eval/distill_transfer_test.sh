#!/usr/bin/env bash
# Does the foveated LoRA transfer to BFL's ALREADY-DISTILLED klein-4B?
#
# klein-4B is klein-base-4B after BFL ran LADD on it: step-distilled to 4 steps AND
# guidance-distilled (no CFG). Verified identical architecture -- same 169 tensor names and
# shapes, weights differing only ~3% relative in the transformer blocks -- so our foveated
# LoRA is mechanically loadable on it. If a rank-32 delta learned on the base survives that
# 3% shift, we get a 4-step foveated model with no distillation training at all.
#
# Four arms, one prompt, same seed/mask, so the comparison is interpretable:
#   A base + LoRA @ 50 steps cfg4  -> our current method (quality reference)
#   B base + LoRA @  4 steps cfg4  -> naive few-step of our method (what distillation must beat)
#   C dist + LoRA @  4 steps cfg1  -> THE TEST
#   D dist         @  4 steps cfg1  -> distilled model on mixed-res tokens without our LoRA
#                                     (isolates whether the LoRA contributes anything)
set -uo pipefail
export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True
# /home/brianchc/DiffSynth-Studio is a clean upstream checkout that is pip-installed and
# SHADOWS this repo's DiffSynth-Studio (which is the one carrying the foveated pipeline).
# Without this, `import diffsynth` succeeds but flux2_image_foveated is missing.
export PYTHONPATH="/home/brianchc/foveated_diffusion-private/DiffSynth-Studio:${PYTHONPATH:-}"

GPU=${GPU:-7}
LORA=${LORA:-/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-10000.safetensors}
BASE_ID="black-forest-labs/FLUX.2-klein-base-4B"
DIST_ID="black-forest-labs/FLUX.2-klein-4B"
OUT=${OUT:-/home/brianchc/foveated_diffusion-private/results/distill_transfer_test}
# the lion prompt used throughout this codebase (baseline_qual.sh / naive_qual.sh / vary_radius.sh)
PROMPT="A lion lounges calmly, displaying its powerful physique, with a soft focus highlighting its majestic mane and intense gaze."

cd "$(dirname "$0")"
mkdir -p "$OUT"

run () {  # run <name> <model_id> <steps> <cfg> <lora|none>
  local name=$1 model=$2 steps=$3 cfg=$4 lora=$5
  local extra=()
  [ "$lora" != "none" ] && extra=(--lora_checkpoint "$lora")
  echo "=============== $name : $model, ${steps} steps, cfg $cfg, lora=$(basename "$lora") ==============="
  CUDA_VISIBLE_DEVICES=$GPU python inference.py \
    --experiment ours \
    --model_id "$model" \
    --num_inference_steps "$steps" \
    --guidance_scale "$cfg" \
    --decode_mode merge \
    --lr_downsample_factor 2 \
    --mask_radius 0.5 \
    --full_eval_mask square \
    --soft_foveation_blend true \
    --foveation_outline false \
    --tokenAE_training false \
    --seed 42 \
    --output_dir "$OUT/$name" \
    --prompt "$PROMPT" "${extra[@]}" 2>&1 | tail -4
  echo "--> $OUT/$name"
}

run A_base50_lora   "$BASE_ID" 50 4.0 "$LORA"
run B_base4_lora    "$BASE_ID"  4 4.0 "$LORA"
run C_dist4_lora    "$DIST_ID"  4 1.0 "$LORA"
run D_dist4_nolora  "$DIST_ID"  4 1.0 none

echo; echo "=== results ==="; find "$OUT" -name "*.png" | sort
