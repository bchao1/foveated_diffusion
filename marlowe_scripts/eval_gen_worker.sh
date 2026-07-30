#!/bin/bash
# =====================================================================================
# Generation worker, shard mode: ONE array element = ONE shard = ONE GPU = one preempt
# allocation. Follows marlowe_scripts/marlowe_launch_rules.md.
#
# Submitted by submit_eval.sh; not run by hand. Required env (passed via --export):
#   METHOD       high_res | naive | ours | tome | ralu
#   MASK_CONFIG  <shape>:<radius>, e.g. circular:0.25
#   NUM_SHARDS   array size (default 8)
#
# The shard split is deterministic — element i takes the prompts where
# global_idx % NUM_SHARDS == i — and the driver skips any sample whose manifest row AND
# image already exist, so a requeue after preemption resumes instead of restarting.
# See eval/shard_manifest.py.
#
# Resource directives are also passed on the sbatch command line by submit_eval.sh
# (GEN_GPUS/GEN_CPUS/GEN_MEM/GEN_TIME), which overrides the values here.
# =====================================================================================
#SBATCH --job-name=fovgen
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -uo pipefail

SHARD_ID="${SLURM_ARRAY_TASK_ID:?submit as an array}"
NUM_SHARDS="${NUM_SHARDS:-8}"
METHOD="${METHOD:?set METHOD}"
MASK_CONFIG="${MASK_CONFIG:?set MASK_CONFIG, e.g. circular:0.25}"

# --export=NIL means nothing came from the submitting shell; rebuild it here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/eval_env.sh"
activate_python || exit 1
setup_node_scratch          # node-local Triton/Inductor caches + per-job MPS dirs
mkdir -p "$LOG_DIR"

OUT_DIR=$(eval_out_dir "${METHOD}" "${MASK_CONFIG}")

echo "=============================================================="
echo " ${METHOD}  ${MASK_CONFIG}  shard ${SHARD_ID}/${NUM_SHARDS}"
echo " job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname), $(date -Is)"
echo " gpus=${SLURM_GPUS_ON_NODE:-?} cpus=${SLURM_CPUS_PER_TASK:-?} out=${OUT_DIR}"
echo "=============================================================="

requeued=0
requeue_self() {
    if [ "$requeued" = 0 ]; then
        requeued=1
        echo "[$(date -Is)] USR1: preemption imminent, requeueing this element only"
        # Requeue only this element, never the array's base job ID.
        target="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        scontrol requeue "$target"
    fi
}
trap requeue_self USR1

# Every generation flag comes from eval/eval.sh, the same script used on local GPUs, so a
# cluster shard and a local shard are the same computation. CUDA_VISIBLE_DEVICES is left to
# Slurm: `eval.sh shard` does not override it.
export_eval_env
"${REPO}/eval/eval.sh" shard "${METHOD}" "${MASK_CONFIG}" "${SHARD_ID}" "${NUM_SHARDS}" &
child=$!
wait "$child"
rc=$?

[ "$requeued" = 1 ] && exit 0
if [ "$rc" -ne 0 ]; then
    echo "[$(date -Is)] driver exited ${rc}"
    exit "$rc"
fi

# The driver writes .generation_shard<id>_complete itself, only once its shard has no
# pending samples. Report it so the log says plainly whether this element is finished.
if [ -f "${OUT_DIR}/.generation_shard${SHARD_ID}_complete" ]; then
    echo "[$(date -Is)] shard ${SHARD_ID} complete: $(ls "${OUT_DIR}"/img_*.png 2>/dev/null | wc -l) images in ${OUT_DIR}"
else
    echo "[$(date -Is)] WARNING: no completion marker in ${OUT_DIR}" >&2
fi
exit 0
