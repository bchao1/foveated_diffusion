#!/bin/bash
# =====================================================================================
# Launcher for the Marlowe eval sweep. Generation only — no scoring job is submitted; run
# the metric scripts on the output directories afterwards (see MARLOWE_EVAL.md).
#
# Two submission shapes, both on the preempt partition:
#
#   shard mode (marlowe_launch_rules.md)   one array element per shard, 1 GPU each
#   node mode  (example.sh shape)          one job per config, whole node, all shards
#                                          in parallel; 1 job record instead of NUM_SHARDS
#
#   ./submit_eval.sh preflight                    check paths, print the plan
#   ./submit_eval.sh check-env                    actually activate conda and import torch
#   ./submit_eval.sh print                        exact sbatch commands, submit nothing
#   ./submit_eval.sh print-node                   same, node mode
#   ./submit_eval.sh submit ours circular:0.25    one method + config, array
#   ./submit_eval.sh submit ours                  one method, every config, array
#   ./submit_eval.sh submit-all                   every method x config, array
#   ./submit_eval.sh submit-node ours circular:0.25   one whole-node job
#   ./submit_eval.sh submit-node-all              every method x config, node mode
#   ./submit_eval.sh status                       per-shard completion markers
# =====================================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/eval_env.sh"

MODE=${1:-preflight}
SUBMIT=0          # print modes build the identical command lines without submitting

acct_arg=()
[ -n "${ACCOUNT}" ] && acct_arg=(--account "${ACCOUNT}")
mail_arg=()
[ -n "${MAIL_TYPE}" ] && mail_arg=(--mail-type "${MAIL_TYPE}")
[ -n "${MAIL_USER}" ] && mail_arg+=(--mail-user "${MAIL_USER}")

mask_tag() {   # mask_tag <shape:radius>
    local tag="${1%%:*}_r${1##*:}"
    [ "${RANDOM_MASK_CENTER}" = "true" ] && tag="${tag}_randcenter${MASK_CENTER_SEED}"
    echo "$tag"
}

# ---- one (method, config) as an array of single-GPU shards ---------------------------
submit_shards() {   # submit_shards <method> <shape:radius>
    local method="$1" cfg="$2"
    local out; out=$(eval_out_dir "$method" "$cfg")
    local name="fovgen_${method}_$(mask_tag "$cfg")"
    # Only when submitting: the print modes must work from a machine without $ROOT mounted.
    [ "${SUBMIT}" = "1" ] && mkdir -p "$LOG_DIR" "$out"

    # --export=NIL avoids Slurm user-environment retrieval failures; the worker rebuilds
    # what it needs from eval_env.sh. Per-job variables are passed explicitly.
    local cmd=(sbatch --parsable "${acct_arg[@]}" "${mail_arg[@]}"
        --job-name "${name}"
        --partition "${PARTITION}"
        --array "0-$((NUM_SHARDS - 1))"
        --nodes 1 --ntasks-per-node 1
        --gpus-per-node "${GEN_GPUS}" --cpus-per-task "${GEN_CPUS}" --mem "${GEN_MEM}"
        --time "${GEN_TIME}"
        --output "${LOG_DIR}/${name}_%A_%a.out"
        --error  "${LOG_DIR}/${name}_%A_%a.err"
        --export "NIL,METHOD=${method},MASK_CONFIG=${cfg},NUM_SHARDS=${NUM_SHARDS}"
        "${SCRIPT_DIR}/eval_gen_worker.sh")

    if [ "${SUBMIT}" = "0" ]; then
        echo "# ---- ${method}  ${cfg}  (array 0-$((NUM_SHARDS - 1)), ${GEN_GPUS} gpu/element) ----"
        echo "${cmd[*]}"
        echo
        return 0
    fi
    local job; job=$("${cmd[@]}") || { echo "submit failed: ${method} ${cfg}" >&2; return 1; }
    job=${job%%;*}          # strip any ";cluster" suffix
    echo "submitted ${job}  ${method} ${cfg}  array 0-$((NUM_SHARDS - 1))  -> ${out}"
}

# ---- one (method, config) as a single whole-node job ---------------------------------
submit_node() {   # submit_node <method> <shape:radius>
    local method="$1" cfg="$2"
    local out; out=$(eval_out_dir "$method" "$cfg")
    local name="fovnode_${method}_$(mask_tag "$cfg")"
    [ "${SUBMIT}" = "1" ] && mkdir -p "$LOG_DIR" "$out"

    local cmd=(sbatch --parsable "${acct_arg[@]}" "${mail_arg[@]}"
        --job-name "${name}"
        --partition "${PARTITION}"
        --nodes 1 --ntasks-per-node 1
        --gpus-per-node "${NODE_GPUS}" --cpus-per-task "${NODE_CPUS}" --mem "${NODE_MEM}"
        --time "${NODE_TIME}"
        --output "${LOG_DIR}/${name}_%j.out"
        --error  "${LOG_DIR}/${name}_%j.err"
        --export "NIL,METHOD=${method},MASK_CONFIG=${cfg},NUM_SHARDS=${NUM_SHARDS}"
        "${SCRIPT_DIR}/eval_node_worker.sh")

    if [ "${SUBMIT}" = "0" ]; then
        echo "# ---- ${method}  ${cfg}  (1 job, ${NODE_GPUS} gpus, ${NUM_SHARDS} shards) ----"
        echo "${cmd[*]}"
        echo
        return 0
    fi
    local job; job=$("${cmd[@]}") || { echo "submit failed: ${method} ${cfg}" >&2; return 1; }
    job=${job%%;*}
    echo "submitted ${job}  ${method} ${cfg}  ${NUM_SHARDS} shards on ${NODE_GPUS} gpus  -> ${out}"
}

configs_for() {   # configs_for <method>: high_res is mask-independent, so it runs once
    if [ "$1" = "high_res" ]; then echo "${MASK_CONFIGS}" | awk '{print $1}'
    else echo "${MASK_CONFIGS}"; fi
}

for_each_pair() {   # for_each_pair <fn> [method] [config]
    local fn="$1"; shift
    if [ "$#" -ge 2 ]; then "$fn" "$1" "$2"; return $?; fi
    if [ "$#" -eq 1 ]; then
        for cfg in $(configs_for "$1"); do "$fn" "$1" "$cfg"; done
        return 0
    fi
    for m in ${METHODS}; do
        for cfg in $(configs_for "$m"); do "$fn" "$m" "$cfg"; done
    done
}

preflight() {
    echo "======================================================================"
    echo " Marlowe eval preflight — generation only, no scoring job"
    echo "   ROOT        ${ROOT}"
    echo "   REPO        ${REPO}"
    echo "   OUT_ROOT    ${OUT_ROOT}"
    echo "   LOG_DIR     ${LOG_DIR}"
    echo "   partition   ${PARTITION}   account ${ACCOUNT:-<site default>}"
    echo "   shard mode  array 0-$((NUM_SHARDS - 1)), ${GEN_GPUS} gpu / ${GEN_CPUS} cpu / ${GEN_MEM} / ${GEN_TIME} per element"
    echo "   node mode   1 job, ${NODE_GPUS} gpu / ${NODE_CPUS} cpu / ${NODE_MEM} / ${NODE_TIME}, ${NUM_SHARDS} shards"
    echo "   methods     ${METHODS}"
    echo "   configs     ${MASK_CONFIGS}"
    echo "   mask center $([ "${RANDOM_MASK_CENTER}" = "true" ] && echo "random, seed ${MASK_CENTER_SEED}, range +/-${MASK_CENTER_RANGE}, clamped ${CLAMP_MASK_TO_FRAME}" || echo centered)"
    echo "   prompts     ${PROMPTS}${NUM_PROMPTS:+  (capped at ${NUM_PROMPTS})}"
    echo "   conda       ${CONDA_SH}"
    echo "               ${CONDA_ENV}"
    echo "   models      ${DIFFSYNTH_MODEL_BASE_PATH}"
    echo "----------------------------------------------------------------------"
    local pairs=0
    for m in ${METHODS}; do for _ in $(configs_for "$m"); do pairs=$((pairs + 1)); done; done
    echo "   ${pairs} (method, config) pairs"
    echo "     shard mode: ${pairs} x ${NUM_SHARDS} = $((pairs * NUM_SHARDS)) submitted job records"
    echo "     node mode:  ${pairs} submitted job records"
    echo "   check the account limit:  sacctmgr show assoc where user=\$USER format=account,maxsubmit"
    echo "----------------------------------------------------------------------"
    local missing=0
    check() {   # check <label> <path> <kind: d|f|x>
        if { [ "$3" = "d" ] && [ -d "$2" ]; } || { [ "$3" = "f" ] && [ -f "$2" ]; } \
           || { [ "$3" = "x" ] && [ -x "$2" ]; }; then
            printf "   OK      %-14s %s\n" "$1" "$2"
        else
            printf "   MISSING %-14s %s\n" "$1" "$2"; missing=$((missing + 1))
        fi
    }
    check repo        "${REPO}"                          d
    check eval.sh     "${REPO}/eval/eval.sh"             x
    check diffsynth   "${REPO}/DiffSynth-Studio"         d
    check ralu        "${REPO}/eval/RALU/RALU_fov_inference_diffsynth.py" f
    check tome        "${REPO}/eval/importance-token-merge/tome_fov_inference.py" f
    check manifest    "${REPO}/eval/shard_manifest.py"   f
    check prompts     "${PROMPTS}"                       f
    check dataset     "${DATASET_DIR}"                   d
    check models      "${DIFFSYNTH_MODEL_BASE_PATH}"     d
    check nofov_lora  "${NOFOV_LORA}"                    f
    check fov_lora    "${FOV_LORA}"                      f
    check conda_sh    "${CONDA_SH}"                      f
    check conda_env   "${CONDA_ENV}"                     d
    check worker      "${SCRIPT_DIR}/eval_gen_worker.sh" x
    check node_worker "${SCRIPT_DIR}/eval_node_worker.sh" x
    echo "======================================================================"
    if [ "$missing" -gt 0 ]; then
        echo " ${missing} path(s) missing — fix them in eval_env.sh before submitting." >&2
        return 1
    fi
    echo " paths OK. Next:  ./submit_eval.sh check-env   then   ./submit_eval.sh print"
}

check_env() {
    # Activating and importing catches what a path check cannot: a conda env that exists
    # but cannot satisfy the drivers. Run on a login node; no GPU needed.
    echo "activating ${CONDA_ENV} ..."
    activate_python || return 1
    "${PY}" - <<'EOF'
import importlib, sys
ok = True
for mod in ("torch", "diffsynth", "pandas", "PIL", "yaml"):
    try:
        importlib.import_module(mod)
        print(f"  OK      import {mod}")
    except Exception as e:
        ok = False
        print(f"  FAIL    import {mod}: {e}")
print("environment usable" if ok else "environment NOT usable")
sys.exit(0 if ok else 1)
EOF
}

status() {
    printf "%-9s %-26s %-10s %8s  %s\n" METHOD CONFIG SHARDS IMAGES DIR
    for m in ${METHODS}; do
        for cfg in $(configs_for "$m"); do
            local d; d=$(eval_out_dir "$m" "$cfg")
            local done_n; done_n=$(ls "$d"/.generation_shard*_complete 2>/dev/null | wc -l)
            local imgs; imgs=$(ls "$d"/img_*.png 2>/dev/null | wc -l)
            printf "%-9s %-26s %-10s %8s  %s\n" "$m" "$cfg" "${done_n}/${NUM_SHARDS}" "$imgs" "$d"
        done
    done
}

case "${MODE}" in
  preflight) preflight ;;
  check-env) check_env ;;
  print)      shift; for_each_pair submit_shards "$@" ;;
  print-node) shift; for_each_pair submit_node   "$@" ;;
  submit)
    SUBMIT=1; shift
    preflight >/dev/null || { echo "preflight failed; run ./submit_eval.sh preflight" >&2; exit 1; }
    for_each_pair submit_shards "$@"
    ;;
  submit-all)
    SUBMIT=1
    preflight >/dev/null || { echo "preflight failed; run ./submit_eval.sh preflight" >&2; exit 1; }
    for_each_pair submit_shards
    ;;
  submit-node)
    SUBMIT=1; shift
    preflight >/dev/null || { echo "preflight failed; run ./submit_eval.sh preflight" >&2; exit 1; }
    for_each_pair submit_node "$@"
    ;;
  submit-node-all)
    SUBMIT=1
    preflight >/dev/null || { echo "preflight failed; run ./submit_eval.sh preflight" >&2; exit 1; }
    for_each_pair submit_node
    ;;
  status) status ;;
  *)
    echo "usage: $0 {preflight|check-env|print|print-node|submit|submit-all|submit-node|submit-node-all|status}" >&2
    exit 1
    ;;
esac
