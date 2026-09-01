#!/usr/bin/env bash
# Queue the base-model (50 step, cfg 4.0) Run A behind the currently-running distilled
# sweep, on 5 GPUs. Resume-safe: rerun this exact command after any interruption and
# SKIP_COMPLETE + ShardManifest pick up where it stopped.
#
# GPUS must stay "1 2 3 4 5" across every resume: prompt->shard assignment is
# global_idx % NUM_SUBSETS, so changing the GPU count re-shards the prompt set and appends
# duplicate metadata rows for prompts already on disk.
set -u

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_PID=${WAIT_PID:-}
TARGET_GPUS="1 2 3 4 5"
OUT_ROOT=/local/brian/foveated_diffusion/results/iclr_2027_base

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

if [ -n "$WAIT_PID" ]; then
    echo "[$(ts)] waiting for distilled sweep (pid $WAIT_PID) to exit"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(ts)] distilled sweep exited"
fi

# The box is shared. Refuse to launch onto a GPU someone else took while we waited,
# rather than OOM-fighting them halfway through a 5-day run.
echo "[$(ts)] GPU preflight:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
busy=""
for g in $TARGET_GPUS; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    [ "$used" -gt 2000 ] && busy="$busy $g($used MiB)"
done
if [ -n "$busy" ]; then
    echo "[$(ts)] ABORT: target GPUs occupied:$busy"
    echo "         Free them and rerun, or edit TARGET_GPUS -- but keep the count at 5"
    echo "         for the resume to stay consistent with any partial output."
    exit 1
fi

echo "[$(ts)] launching base Run A: 50 steps, cfg 4.0, GPUs $TARGET_GPUS -> $OUT_ROOT"
cd "$EVAL_DIR" || exit 1
MODEL=base \
RANDOM_MASK_CENTER=true \
MASK_CONFIGS="circular:0.1 circular:0.25 circular:0.5" \
GPUS="$TARGET_GPUS" \
OUT_ROOT="$OUT_ROOT" \
./eval.sh all
echo "[$(ts)] base Run A exit=$?"
