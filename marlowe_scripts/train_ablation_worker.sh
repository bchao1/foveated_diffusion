#!/bin/bash
# =====================================================================================
# sbatch body for one ablation training run. Submitted by submit_ablation.sh, which sets
# partition, account, GPUs, time and log paths on the sbatch command line.
#   Required env: RUN   abl_rope_standard | abl_res_embed | smoke_abl_*
# A smoke run trains 20 steps on one GPU and checks the saved checkpoint.
# =====================================================================================
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
set -uo pipefail

RUN="${RUN:?set RUN}"
SCRIPT_DIR="${SCRIPT_DIR:?submit_ablation.sh sets SCRIPT_DIR}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/ablation_env.sh"
activate_python || exit 1
setup_node_scratch

NGPU=$(nvidia-smi -L | wc -l)
if [[ "$RUN" == smoke_* ]]; then NPROC=1; else NPROC=${TRAIN_PROCS}; fi
[ "$NPROC" -le "$NGPU" ] || { echo "need ${NPROC} GPUs, node has ${NGPU}" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC - 1)))
if [[ "$RUN" == smoke_* ]]; then
    OUT="${EXP_ROOT}/smoke/${RUN}_${SLURM_JOB_ID}"
else
    OUT="${EXP_ROOT}/${RUN}"
fi
mkdir -p "$OUT"
mapfile -t ARGS < <(train_args "$RUN" "$OUT") || exit 1

echo "=============================================================="
echo " ${RUN}  job ${SLURM_JOB_ID} on $(hostname), $(date -Is)"
echo " gpus on node=${NGPU}  processes=${NPROC} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})  out=${OUT}"
echo " repo $(git -C "${REPO}" rev-parse HEAD) ($(git -C "${REPO}" status --porcelain | wc -l) dirty files)"
echo "=============================================================="
{ echo "job ${SLURM_JOB_ID}"; echo "commit $(git -C "${REPO}" rev-parse HEAD)"; printf '%q ' "${ARGS[@]}"; echo; } > "${OUT}/launch_${SLURM_JOB_ID}.txt"

# GPU memory trace (peak memory is the open question for the smoke run).
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 30 > "${OUT}/gpu_${SLURM_JOB_ID}.csv" &
smi_pid=$!

# Effective batch = processes x 1 image. 6 processes reproduce the reference run.
launch=(--num_machines 1 --num_processes "${NPROC}" --mixed_precision bf16 --dynamo_backend no)
[ "$NPROC" -gt 1 ] && launch+=(--multi_gpu)
accelerate launch "${launch[@]}" \
    "${REPO}/DiffSynth-Studio/examples/flux2/model_training/train_foveated.py" "${ARGS[@]}"
rc=$?
kill "$smi_pid" 2>/dev/null
echo "peak GPU memory (MiB): $(awk -F', ' 'NR>1{gsub(/ MiB/,"",$3); if($3>m)m=$3} END{print m}' "${OUT}/gpu_${SLURM_JOB_ID}.csv")"

# The job succeeds only if the final checkpoint exists and describes the right DiT.
last=$(ls "${OUT}"/step-*.safetensors 2>/dev/null | sort -V | tail -1)
if [[ "$RUN" != smoke_* ]] && [ "$last" != "${OUT}/step-${MAX_STEPS}.safetensors" ]; then
    echo "final checkpoint step-${MAX_STEPS}.safetensors missing (last: ${last:-none})" >&2; rc=1
fi
"${PY}" - "$last" "$RUN" <<'EOF' || rc=1
import sys
from diffsynth.models.flux2_dit_foveated import read_foveation_arch
path, run = sys.argv[1], sys.argv[2]
a = read_foveation_arch(path)
want = {"rope_standard": ("standard", False), "res_embed": ("phase_aligned", True)}
key = "rope_standard" if "rope_standard" in run else "res_embed"
got = (a["rope_mode"], a["resolution_embedding"])
print(f"checkpoint {path}: rope_mode={got[0]} resolution_embedding={got[1]} metadata={a['has_metadata']}")
if a["embedding"] is not None:
    print("  embedding |row| LR/HR:", a["embedding"].float().norm(dim=1).tolist())
sys.exit(0 if (a["has_metadata"] and got == want[key]) else 1)
EOF
echo "exit ${rc} at $(date -Is)"
exit $rc
