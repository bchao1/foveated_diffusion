#!/bin/bash
# =====================================================================================
# Generation worker, whole-node mode: ONE job takes a whole node and runs every shard of
# one (method, mask config) in parallel, one shard per GPU — the resource shape example.sh
# uses, and exactly what eval/eval.sh does on local GPUs.
#
# Trade-off against the array mode in eval_gen_worker.sh:
#   + 1 submitted-job record per config instead of NUM_SHARDS (21 vs 189 for a full sweep)
#   - coarser requeue: preemption takes all shards down together. Each shard still resumes
#     from its own manifest, so no generated image is ever lost or redone.
#
# Submitted by submit_eval.sh (`submit-node`); not run by hand. Required env:
#   METHOD       high_res | naive | ours | tome | ralu
#   MASK_CONFIG  <shape>:<radius>
#   NUM_SHARDS   shards to run in parallel; must be <= GPUs on the node
# =====================================================================================
#SBATCH --job-name=fovnode
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail

NUM_SHARDS="${NUM_SHARDS:-8}"
METHOD="${METHOD:?set METHOD}"
MASK_CONFIG="${MASK_CONFIG:?set MASK_CONFIG, e.g. circular:0.25}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/eval_env.sh"
activate_python || exit 1
setup_node_scratch
mkdir -p "$LOG_DIR"

OUT_DIR=$(eval_out_dir "${METHOD}" "${MASK_CONFIG}")

echo "=============================================================="
echo " ${METHOD}  ${MASK_CONFIG}  ${NUM_SHARDS} shards on one node"
echo " job ${SLURM_JOB_ID} on $(hostname), $(date -Is)"
echo " gpus=${SLURM_GPUS_ON_NODE:-?} cpus=${SLURM_CPUS_PER_TASK:-?} out=${OUT_DIR}"
echo "=============================================================="

pids=()
requeued=0
requeue_self() {
    if [ "$requeued" = 0 ]; then
        requeued=1
        echo "[$(date -Is)] USR1: preemption imminent, requeueing this job"
        scontrol requeue "${SLURM_JOB_ID}"
    fi
}
trap requeue_self USR1

# One `eval.sh shard` per GPU. NODE_GPU_IDS defaults to 0..NUM_SHARDS-1, which is right when
# the job owns the whole node; override it to pin specific devices (e.g. for a local test).
gpu_ids=(${NODE_GPU_IDS:-$(seq 0 $((NUM_SHARDS - 1)))})
[ "${#gpu_ids[@]}" -ge "$NUM_SHARDS" ] || {
    echo "NODE_GPU_IDS has ${#gpu_ids[@]} entries but NUM_SHARDS=${NUM_SHARDS}" >&2; exit 2; }

export_eval_env
tag="${MASK_CONFIG//:/}"
for ((i = 0; i < NUM_SHARDS; i++)); do
    CUDA_VISIBLE_DEVICES="${gpu_ids[$i]}" \
        "${REPO}/eval/eval.sh" shard "${METHOD}" "${MASK_CONFIG}" "$i" "${NUM_SHARDS}" \
        > "${LOG_DIR}/node_${METHOD}_${tag}_${SLURM_JOB_ID}_shard${i}.log" 2>&1 &
    pids+=($!)
done

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=$?
done

[ "$requeued" = 1 ] && exit 0
if [ "$rc" -ne 0 ]; then
    echo "[$(date -Is)] at least one shard exited non-zero (last rc=${rc}); see per-shard logs" >&2
    exit "$rc"
fi

done_n=$(ls "${OUT_DIR}"/.generation_shard*_complete 2>/dev/null | wc -l)
echo "[$(date -Is)] ${done_n}/${NUM_SHARDS} shards complete, " \
     "$(ls "${OUT_DIR}"/img_*.png 2>/dev/null | wc -l) images in ${OUT_DIR}"
[ "$done_n" -eq "$NUM_SHARDS" ] || { echo "WARNING: shards still pending" >&2; }
exit 0
