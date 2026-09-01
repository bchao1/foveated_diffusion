#!/usr/bin/env bash
# Fill in whatever the base Run A sweep lost to transient failures (a neighbour grabbing a
# GPU mid-run, etc). Idempotent: SKIP_COMPLETE skips every (method, config) already at
# 10000 pngs, and ShardManifest resumes a partial one sample-by-sample.
#
# Must stay on the SAME 5 GPUs as the original run -- queue_base_runA.sh owns that decision,
# so this just calls it rather than duplicating the env.
set -u

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_PID=${WAIT_PID:-}
PASSES=${PASSES:-3}
OUT_ROOT=/local/brian/foveated_diffusion/results/iclr_2027_base

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Count (method, config) dirs still short of 10000. Excludes stray dirs like
# .ipynb_checkpoints, which hold no real output but do match a naive recursive glob.
short() {
    local n=0
    for d in "$OUT_ROOT"/high_res "$OUT_ROOT"/*_randcenter*/*; do
        [ -d "$d" ] || continue
        case "$d" in *.ipynb_checkpoints*|*/logs) continue;; esac
        local have; have=$(find "$d" -maxdepth 1 -name 'img_*.png' 2>/dev/null | wc -l)
        [ "$have" -lt 10000 ] && { echo "   short: ${d#$OUT_ROOT/}  $have/10000"; n=$((n+1)); }
    done
    return $n
}

if [ -n "$WAIT_PID" ]; then
    echo "[$(ts)] waiting for base Run A (pid $WAIT_PID) to exit"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
    echo "[$(ts)] base Run A exited"
fi

for pass in $(seq 1 "$PASSES"); do
    echo "[$(ts)] ---- repair pass $pass/$PASSES ----"
    if short; then
        echo "[$(ts)] all dirs at 10000/10000 -- nothing to repair"
        exit 0
    fi
    "$EVAL_DIR"/queue_base_runA.sh
    echo "[$(ts)] pass $pass finished (exit $?)"
done

echo "[$(ts)] ---- after $PASSES passes ----"
if short; then
    echo "[$(ts)] COMPLETE: 130000/130000"
else
    echo "[$(ts)] STILL SHORT after $PASSES passes -- check $OUT_ROOT/logs for a persistent failure"
    exit 1
fi
