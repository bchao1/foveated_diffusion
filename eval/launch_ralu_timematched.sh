#!/usr/bin/env bash
# Wall-clock-matched RALU baseline, full scale: 3 radii x 10000 prompts, base model.
#
# Each radius gets its own explicit RALU_N, solved so RALU's mean (dit + vae) per image
# equals `ours` at the same radius. eval.sh takes one RALU_N per invocation, hence one run
# per radius rather than one run with three MASK_CONFIGS.
#
# Validated 2026-08-12 on GPUs 0 2 3 4 5, 5 shards, 50 prompts/arm, idle box
# (results/iclr_2027_timematch_cal2). Measured ralu/ours, 95% CI +/-0.4%:
#     r=0.1  N=[11,12,13]  36 steps  ratio 1.010
#     r=0.25 N=[11,12,15]  38 steps  ratio 1.016
#     r=0.5  N=[15,16,19]  50 steps  ratio 1.029
# All residuals give RALU slightly MORE time than ours, i.e. the match is conservative
# with respect to any RALU quality deficit. Accepted deliberately.
#
# Note: these N are NOT compute-matched. Holding wall-clock equal hands RALU ~1.1-1.4x the
# token-steps of ours, because RALU's advantage is spending equal FLOPs in fewer, larger
# steps. The compute-matched arm lives in results/iclr_2027_base/*/ralu; keep both.
#
# RESUME: rerun this script verbatim. SKIP_COMPLETE skips a radius already at 10000 pngs and
# ShardManifest resumes a partial one. GPUS must stay "0 2 3 4 5" -- prompt->shard is
# global_idx % NUM_SUBSETS, so a different GPU count re-shards and duplicates metadata rows.
set -u

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_GPUS=${TARGET_GPUS:-"0 2 3 4 5"}
OUT_ROOT=${OUT_ROOT:-/local/brian/foveated_diffusion/results/iclr_2027_base_ralu_timematched}

CONFIGS=(
  "0.1|11 12 13"
  "0.25|11 12 15"
  "0.5|15 16 19"
)

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[$(ts)] GPU preflight (target: $TARGET_GPUS)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
busy=""
for g in $TARGET_GPUS; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    [ "$used" -gt 2000 ] && busy="$busy $g(${used}MiB)"
done
if [ -n "$busy" ]; then
    echo "[$(ts)] ABORT: target GPUs occupied:$busy"
    echo "         Free them and rerun. Keep the GPU COUNT at 5 so the resume stays"
    echo "         consistent with any partial output already on disk."
    exit 1
fi

for entry in "${CONFIGS[@]}"; do
    r="${entry%%|*}"; N="${entry##*|}"
    echo "[$(ts)] ===== r=$r  RALU_N=[$N] ====="
    MODEL=base \
    RANDOM_MASK_CENTER=true \
    MASK_CONFIGS="circular:$r" \
    RALU_N="$N" \
    GPUS="$TARGET_GPUS" \
    OUT_ROOT="$OUT_ROOT" \
    "$EVAL_DIR"/eval.sh ralu
    echo "[$(ts)] r=$r exit=$?"
done

echo "[$(ts)] ===== final counts ====="
for entry in "${CONFIGS[@]}"; do
    r="${entry%%|*}"
    d="$OUT_ROOT/circular_r${r}_randcenter42/ralu"
    echo "  r=$r  $(find "$d" -maxdepth 1 -name 'img_*.png' 2>/dev/null | wc -l)/10000"
done
echo "[$(ts)] done"
