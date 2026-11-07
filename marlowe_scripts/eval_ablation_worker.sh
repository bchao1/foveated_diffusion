#!/bin/bash
# =====================================================================================
# sbatch body: Table 1 eval of one LoRA (ours arm, circular:0.25, random centers seed 42,
# 10k metadata_test.csv prompts, base model, 50 steps, CFG 4.0), 8 shards on one node.
# Submitted by submit_ablation.sh. Required env: ARM, SCRIPT_DIR.
# =====================================================================================
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
set -uo pipefail
ARM="${ARM:?set ARM}"
SCRIPT_DIR="${SCRIPT_DIR:?submit_ablation.sh sets SCRIPT_DIR}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/ablation_env.sh"
activate_python || exit 1
setup_node_scratch

export FOV_LORA; FOV_LORA=$(eval_lora "$ARM") || exit 1
[ -f "$FOV_LORA" ] || { echo "missing checkpoint ${FOV_LORA}" >&2; exit 1; }
export OUT_ROOT="${RESULTS_ROOT}/${ARM}" LOG_DIR="${RESULTS_ROOT}/${ARM}/logs"
export PROMPTS="${EVAL_PROMPTS}" METHODS=ours MASK_CONFIGS="${EVAL_MASK}"
export RANDOM_MASK_CENTER=true MASK_CENTER_SEED=42 MODEL=base PY
mkdir -p "$LOG_DIR"
NSH=$(nvidia-smi -L | wc -l)
echo "eval ${ARM}  job ${SLURM_JOB_ID} on $(hostname)  lora=${FOV_LORA}  shards=${NSH}  out=${OUT_ROOT}"
"${PY}" -c "from diffsynth.models.flux2_dit_foveated import read_foveation_arch as r; a=r('${FOV_LORA}'); print('arch', a['rope_mode'], a['resolution_embedding'], 'metadata', a['has_metadata'])"

pids=()
for ((i = 0; i < NSH; i++)); do
    CUDA_VISIBLE_DEVICES=$i "${REPO}/eval/eval.sh" shard ours "${EVAL_MASK}" "$i" "$NSH" \
        > "${LOG_DIR}/shard${i}_${SLURM_JOB_ID}.log" 2>&1 &
    pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
"${REPO}/eval/eval.sh" check 2>/dev/null | tail -2
n=$(ls "${OUT_ROOT}"/circular_r0.25_randcenter42/ours/img_*.png 2>/dev/null | wc -l)
echo "pngs: ${n}/10000"; [ "$n" -eq 10000 ] || rc=1
exit $rc
