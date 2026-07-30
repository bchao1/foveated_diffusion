Saving Batch Quota by Running Eval on Preempt


Use this pattern for image generation, evaluation, or other workloads where samples are independent. On our cluster it moves interruptible work off the batch partition, preserving batch quota and using lower-cost preempt capacity.

🔄 Pattern

1. Split the dataset deterministically across an array:

my_indices = [i for i in range(n) if i % num_shards == shard_id]

1. Give each array element one GPU and a one-hour preempt allocation.
2. Save each result immediately. Maintain a per-shard manifest and flush one row only after its output is complete.
3. On every start or requeue, scan the manifest and output files and skip completed samples. Repeating the one interrupted sample is harmless.
4. Submit scoring with afterok so it starts only after every generation shard completes.

This works well because generation is embarrassingly parallel and the model only needs to be reloaded after preemption. It is not intended for a tightly coupled multi-GPU task.

🛠️ Worker Template

#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --output=/path/to/logs/%x_%A_%a.out
#SBATCH --error=/path/to/logs/%x_%A_%a.err
set -uo pipefail
SHARD_ID="${SLURM_ARRAY_TASK_ID:?submit as an array}"
NUM_SHARDS="${NUM_SHARDS:-8}"
# Keep compilation caches node-local and disposable.
export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor_cache"
requeued=0
requeue_self() {
    if [ "$requeued" = 0 ]; then
        requeued=1
        # Requeue only this element, never the array's base job ID.
        target="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        scontrol requeue "$target"
    fi
}
trap requeue_self USR1
python generate.py \
    --shard-id "$SHARD_ID" \
    --num-shards "$NUM_SHARDS" &
child=$!
wait "$child"
rc=$?
[ "$requeued" = 1 ] && exit 0
exit "$rc"

generate.py must be idempotent: use a stable sample index, seed, filename, and configuration; skip a sample only when both its manifest row and final output exist. Prefer per-shard manifests such as `prompts_rank3.csv` to avoid concurrent writes. Merge them atomically under a file lock when needed.

🚀 Launcher Template

GEN_JOB=$(sbatch --parsable --export=NIL \
    --array=0-7 \
    --partition=preempt \
    worker.sh)
GEN_JOB=${GEN_JOB%%;*}
sbatch --export=NIL \
    --partition=preempt \
    --dependency="afterok:${GEN_JOB}" \
    score.sh /path/to/eval-output

`--export=NIL` avoids Slurm user-environment retrieval failures; the worker should reconstruct the few required environment variables explicitly.

⚠️ Practical Cautions

* An eight-element array plus one scoring job consumes nine submitted-job records. Check the account's submitted-job limit before launching multiple sweeps.
* Requeue ${SLURM_ARRAY_JOB_ID}${SLURM_ARRAY_TASK_ID}. Requeueing the base array ID can restart every shard and hit the launch-failure limit.

- Use a unique output directory for each model/configuration. Never resume a timed run from untimed or differently configured manifests.
- Append Slurm logs and write a .generation_shard<id>_complete marker only after that shard has no pending samples.

* Make the scorer resumable too; metric evaluation can also exceed one hour.

