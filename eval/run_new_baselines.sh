#!/bin/bash
# Single entry point for the two new baseline rows (ToMe-2x2 and RALU-foveated) of the
# supplementary baseline figures paper_figures/supp/baseline_{1,2,3,101}.png.
#
#   ./run_new_baselines.sh              # verify, plan, then run both arms over all 3 sets
#   ./run_new_baselines.sh verify       # mask-replay check only, no GPU
#   ./run_new_baselines.sh plan         # print token budgets and RALU schedules, no GPU
#   ./run_new_baselines.sh tome         # ToMe arm only
#   ./run_new_baselines.sh ralu         # RALU arm only
#
#   GPUS="0 6 7" SETS="1 2" ./run_new_baselines.sh
#
# Both arms are compute-matched to ours at 50 steps on the *same per-prompt mask* the
# existing high_res/naive/ours tiles used (circular r=0.33, fixation point replayed from
# run_user_study's seed-42 RNG). See new_baselines_inference.py for the details.
set -uo pipefail

MODE=${1:-all}

export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/miele/brian/modelscope}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-True}
# The imp_tome env resolves `diffsynth` to /home/brianchc/DiffSynth-Studio, a different
# checkout without the foveated pipeline. Force the private repo's copy.
export PYTHONPATH=${PYTHONPATH:-/home/brianchc/foveated_diffusion-private/DiffSynth-Studio}

PY=${PY:-/home/brianchc/miniconda3/envs/imp_tome/bin/python}
SCRIPT=${SCRIPT:-new_baselines_inference.py}

GPUS=${GPUS:-"0 6 7 8 9"}                       # free A6000s at time of writing
SETS=${SETS:-"1 2 3"}
NUM_PROMPTS=${NUM_PROMPTS:-100}
OUT=${OUT:-../results/eccv2026_final_v2}
LOG_DIR=${LOG_DIR:-../results/eccv2026_final_v2/logs_new_baselines}

# Base model on purpose: run_user_study clears LoRA for the high_res and naive columns
# these rows sit next to. Set LORA=<path> to override.
LORA=${LORA:-}

STEPS=${STEPS:-50}
SEED=${SEED:-42}
CFG=${CFG:-4.0}
LR_FACTOR=${LR_FACTOR:-2}
MERGE_MLP=${MERGE_MLP:-true}                    # headline ToMe config (see RUN.md)
RALU_STAGES=${RALU_STAGES:-3}
RALU_LEVEL=${RALU_LEVEL:-4}
Z_MAX=${Z_MAX:-6.5}                             # RALU authors' re-noising strength
OURS_STEPS=${OURS_STEPS:-50}

cd "$(dirname "$0")"
mkdir -p "$LOG_DIR"

COMMON=(--prompt_sets $SETS --num_prompts "$NUM_PROMPTS" --out_root "$OUT"
        --steps "$STEPS" --seed "$SEED" --cfg "$CFG" --lr_downsample_factor "$LR_FACTOR"
        --merge_mlp "$MERGE_MLP" --num_stages "$RALU_STAGES" --ralu_level "$RALU_LEVEL"
        --z_max "$Z_MAX" --ours_steps "$OURS_STEPS")
if [ -n "$LORA" ]; then COMMON+=(--lora_checkpoint "$LORA"); fi

gpu_array=($GPUS)
NUM_SUBSETS=${#gpu_array[@]}

check_gpus() {
    for gpu in "${gpu_array[@]}"; do
        used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "$used" -gt 1024 ]; then
            echo "WARNING: GPU $gpu already has ${used} MiB in use" >&2
        fi
    done
}

run_arm() {
    local arm=$1
    echo "=== $arm : $NUM_SUBSETS shards over GPUs $GPUS ==="
    local pids=()
    for i in "${!gpu_array[@]}"; do
        local gpu=${gpu_array[$i]}
        local log="$LOG_DIR/${arm}_shard${i}_gpu${gpu}.log"
        echo "  shard $i -> GPU $gpu  ($log)"
        CUDA_VISIBLE_DEVICES=$gpu $PY "$SCRIPT" --arm "$arm" \
            "${COMMON[@]}" --num_subsets "$NUM_SUBSETS" --subset_idx "$i" \
            >"$log" 2>&1 &
        pids+=($!)
    done
    local rc=0
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
    if [ $rc -ne 0 ]; then
        echo "!!! $arm: at least one shard failed, see $LOG_DIR" >&2
    fi
    echo "=== $arm done (exit $rc) ==="
    return $rc
}

case "$MODE" in
    verify)
        $PY "$SCRIPT" --verify_centers --prompt_sets $SETS --num_prompts "$NUM_PROMPTS" \
            --reference_root "$OUT"
        ;;
    plan)
        $PY "$SCRIPT" --plan --arm both "${COMMON[@]}"
        ;;
    tome|ralu)
        check_gpus
        run_arm "$MODE"
        ;;
    all)
        $PY "$SCRIPT" --verify_centers --prompt_sets $SETS --num_prompts "$NUM_PROMPTS" \
            --reference_root "$OUT" || exit 1
        $PY "$SCRIPT" --plan --arm both "${COMMON[@]}"
        check_gpus
        run_arm tome
        run_arm ralu
        ;;
    *)
        echo "unknown mode: $MODE (expected: all | tome | ralu | plan | verify)" >&2
        exit 1
        ;;
esac
