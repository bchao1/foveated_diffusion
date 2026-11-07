#!/bin/bash
# =====================================================================================
# Launcher for the ICLR 2027 architecture-ablation training runs on Marlowe.
#
#   ./submit_ablation.sh preflight              paths, env, data, models; submits nothing
#   ./submit_ablation.sh test-only              sbatch --test-only for every job below
#   ./submit_ablation.sh smoke                  1-GPU preempt job: 20 steps per ablation
#   ./submit_ablation.sh submit [run ...]       8-GPU batch job per run (default: both)
#   ./submit_ablation.sh submit-all             both trainings + Table 1 eval of the reference
#                                               now + eval of each ablation (afterok its training)
#   ./submit_ablation.sh status                 squeue + latest checkpoint per run
#
# Runs: abl_rope_standard  abl_res_embed   (definitions in ablation_env.sh)
# =====================================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/ablation_env.sh"
command -v sbatch >/dev/null || module load slurm/slurm/25.05.2 >/dev/null 2>&1
MODE=${1:-preflight}; shift || true
RUNS=(abl_rope_standard abl_res_embed)

sbatch_cmd() {   # sbatch_cmd <run|eval_ARM> [--test-only] [--dependency=...]
    local run="$1" t="${2:-}" dep="${3:-}" name="fovabl_${1}"
    local part acct gpus time worker="train_ablation_worker.sh" exp="RUN=${1}"
    if [[ "$run" == smoke_* ]]; then
        part=preempt; acct="${BASIC_ACCOUNT}"; gpus=1; time="${SMOKE_TIME}"
    elif [[ "$run" == eval_* ]]; then
        part=batch; acct="${CYCLE_ACCOUNT}"; gpus="${TRAIN_GPUS}"; time="${EVAL_TIME}"
        worker="eval_ablation_worker.sh"; exp="ARM=${1#eval_}"
    else
        part=batch; acct="${CYCLE_ACCOUNT}"; gpus="${TRAIN_GPUS}"; time="${TRAIN_TIME}"
    fi
    echo sbatch --parsable ${t} ${dep} --job-name "${name}" --account "${acct}" --partition "${part}" \
        --nodes 1 --ntasks-per-node 1 --gpus-per-node "${gpus}" \
        --cpus-per-task $((8 * gpus)) --mem $((128 * gpus))G --time "${time}" \
        --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" \
        --export "NIL,${exp},SCRIPT_DIR=${SCRIPT_DIR}" \
        "${SCRIPT_DIR}/${worker}"
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
    local ntar; ntar=$(ls "${DATASET_DIR}"/images/*/.done 2>/dev/null | wc -l)
    if [ "$ntar" -eq 106 ]; then echo "  ok      images: 106/106 tars extracted"
    else echo "  MISSING images: ${ntar}/106 tars extracted (see ${DATASET_DIR}/fetch.log)"; ok=0; fi
    grep -q "rope_mode" "${REPO}/DiffSynth-Studio/examples/flux2/model_training/train_foveated.py" \
        && echo "  ok      train_foveated.py has --rope_mode" || { echo "  MISSING --rope_mode: git pull in ${REPO}"; ok=0; }
    for r in "${RUNS[@]}"; do [ -e "${EXP_ROOT}/${r}" ] && echo "  NOTE    ${EXP_ROOT}/${r} exists"; done
    [ "$ok" = 1 ] && echo "preflight OK" || { echo "preflight FAILED"; return 1; }
}

case "$MODE" in
    preflight) preflight ;;
    test-only)
        for r in smoke_abl_rope_standard "${RUNS[@]}" eval_ref_rank32fov_10k; do eval "$(sbatch_cmd "$r" --test-only)"; done ;;
    submit-all)
        preflight || exit 1; mkdir -p "${LOG_ROOT}" "${RESULTS_ROOT}"
        jobs="${EXP_ROOT}/jobs.tsv"
        [ -f "$REF_LORA" ] || { echo "missing ${REF_LORA}"; exit 1; }
        j=$(eval "$(sbatch_cmd eval_ref_rank32fov_10k)") || exit 1; j=${j%%;*}
        printf '%s\t%s\t%s\n' "$(date -Is)" "$j" eval_ref_rank32fov_10k | tee -a "$jobs"
        for r in "${RUNS[@]}"; do
            if ls "${EXP_ROOT}/${r}"/step-*.safetensors >/dev/null 2>&1; then
                echo "skip $r: ${EXP_ROOT}/${r} already has checkpoints"; continue; fi
            t=$(eval "$(sbatch_cmd "$r")") || exit 1; t=${t%%;*}
            printf '%s\t%s\t%s\n' "$(date -Is)" "$t" "$r" | tee -a "$jobs"
            e=$(eval "$(sbatch_cmd "eval_$r" "" "--dependency=afterok:${t}")") || exit 1; e=${e%%;*}
            printf '%s\t%s\t%s\n' "$(date -Is)" "$e" "eval_$r (afterok:$t)" | tee -a "$jobs"
        done ;;
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
        squeue -u "$USER" -o "%.12i %.34j %.9P %.8T %.10M %.12l %.20S %R" | grep -E "JOBID|fovabl" || true
        for r in "${RUNS[@]}"; do
            echo "$r: $(ls "${EXP_ROOT}/${r}"/step-*.safetensors 2>/dev/null | sort -V | tail -1 || echo none)"; done ;;
    *) sed -n 2,12p "$0"; exit 2 ;;
esac
