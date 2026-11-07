#!/bin/bash
# =====================================================================================
# Launcher for the ICLR 2027 architecture-ablation training runs on Marlowe.
#
#   ./submit_ablation.sh preflight              paths, env, data, models; submits nothing
#   ./submit_ablation.sh test-only              sbatch --test-only for every job below
#   ./submit_ablation.sh smoke                  1-GPU preempt job: 20 steps per ablation
#   ./submit_ablation.sh submit [run ...]       8-GPU batch job per run (default: both)
#   ./submit_ablation.sh status                 squeue + latest checkpoint per run
#
# Runs: abl_rope_standard  abl_res_embed   (definitions in ablation_env.sh)
# =====================================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/ablation_env.sh"
MODE=${1:-preflight}; shift || true
RUNS=(abl_rope_standard abl_res_embed)

sbatch_cmd() {   # sbatch_cmd <run> [--test-only]
    local run="$1" t="${2:-}" name="fovabl_${1}"
    local part acct gpus time
    if [[ "$run" == smoke_* ]]; then
        part=preempt; acct="${BASIC_ACCOUNT}"; gpus=1; time="${SMOKE_TIME}"
    else
        part=batch; acct="${CYCLE_ACCOUNT}"; gpus="${TRAIN_GPUS}"; time="${TRAIN_TIME}"
    fi
    echo sbatch --parsable ${t} --job-name "${name}" --account "${acct}" --partition "${part}" \
        --nodes 1 --ntasks-per-node 1 --gpus-per-node "${gpus}" \
        --cpus-per-task $((8 * gpus)) --mem $((128 * gpus))G --time "${time}" \
        --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" \
        --export "NIL,RUN=${run},SCRIPT_DIR=${SCRIPT_DIR}" \
        "${SCRIPT_DIR}/train_ablation_worker.sh"
}

preflight() {
    echo "account  batch=${CYCLE_ACCOUNT}  preempt=${BASIC_ACCOUNT}"
    echo "repo     ${REPO} @ $(git -C "${REPO}" rev-parse --short HEAD) ($(git -C "${REPO}" rev-parse --abbrev-ref HEAD))"
    local ok=1
    for p in "${CONDA_ENV}" "${DATASET_DIR}" "${TRAIN_CSV}" \
             "${DIFFSYNTH_MODEL_BASE_PATH}/black-forest-labs/FLUX.2-klein-base-4B/transformer" \
             "${DIFFSYNTH_MODEL_BASE_PATH}/black-forest-labs/FLUX.2-klein-base-4B/text_encoder" \
             "${DIFFSYNTH_MODEL_BASE_PATH}/black-forest-labs/FLUX.2-klein-base-4B/vae" \
             "${DIFFSYNTH_MODEL_BASE_PATH}/black-forest-labs/FLUX.2-klein-base-4B/tokenizer"; do
        if [ -e "$p" ]; then echo "  ok      $p"; else echo "  MISSING $p"; ok=0; fi
    done
    [ -f "${TRAIN_CSV}" ] && echo "  train rows: $(( $(wc -l < "${TRAIN_CSV}") - 1 )) (expect 95288)"
    grep -q "rope_mode" "${REPO}/DiffSynth-Studio/examples/flux2/model_training/train_foveated.py" \
        && echo "  ok      train_foveated.py has --rope_mode" || { echo "  MISSING --rope_mode: git pull in ${REPO}"; ok=0; }
    for r in "${RUNS[@]}"; do [ -e "${EXP_ROOT}/${r}" ] && echo "  NOTE    ${EXP_ROOT}/${r} exists"; done
    [ "$ok" = 1 ] && echo "preflight OK" || { echo "preflight FAILED"; return 1; }
}

case "$MODE" in
    preflight) preflight ;;
    test-only)
        for r in smoke_abl_rope_standard "${RUNS[@]}"; do eval "$(sbatch_cmd "$r" --test-only)"; done ;;
    smoke)
        preflight || exit 1; mkdir -p "${LOG_ROOT}"
        for r in smoke_abl_rope_standard smoke_abl_res_embed; do
            echo "submitted $(eval "$(sbatch_cmd "$r")")  $r"; done ;;
    submit)
        preflight || exit 1; mkdir -p "${LOG_ROOT}"
        [ "$#" -gt 0 ] && RUNS=("$@")
        for r in "${RUNS[@]}"; do
            if ls "${EXP_ROOT}/${r}"/step-*.safetensors >/dev/null 2>&1; then
                echo "skip $r: ${EXP_ROOT}/${r} already has checkpoints"; continue; fi
            echo "submitted $(eval "$(sbatch_cmd "$r")")  $r"
        done ;;
    status)
        squeue -u "$USER" -o "%.12i %.28j %.9P %.8T %.10M %.12l %R" | grep -E "JOBID|fovabl" || true
        for r in "${RUNS[@]}"; do
            echo "$r: $(ls "${EXP_ROOT}/${r}"/step-*.safetensors 2>/dev/null | sort -V | tail -1 || echo none)"; done ;;
    *) sed -n 2,12p "$0"; exit 2 ;;
esac
