#!/bin/bash
# =====================================================================================
# Marlowe environment for the eval sweep. Sourced by eval_gen_worker.sh,
# eval_node_worker.sh and submit_eval.sh.
#
# Jobs are submitted with --export=NIL (per marlowe_launch_rules.md), so nothing is
# inherited from the submitting shell: everything a worker needs is reconstructed here.
# Values below follow example.sh, which is known to work on this cluster.
#
# Generation only — no scoring job is submitted. Run the metric scripts on the output
# directories afterwards (see MARLOWE_EVAL.md).
# =====================================================================================

# ---- scratch root and repo -----------------------------------------------------------
ROOT=${ROOT:-/scratch/m000051-pm05/brianchc_2}
REPO=${REPO:-/users/brianchc/foveated_diffusion}

# ---- python environment (example.sh) -------------------------------------------------
# Reconstructed explicitly because of --export=NIL. CONDA_ENV is a prefix *path*, which is
# what `conda activate` receives in example.sh.
CONDA_SH=${CONDA_SH:-/cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/scratch/m000051/brian/conda_environments/camera_ctrl}
PY=${PY:-}                          # empty = use `python` from the activated env

# ---- model cache (example.sh) -------------------------------------------------------
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT}/models}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-True}
# The eval drivers import `diffsynth` from this repo's checkout, not a site-packages copy.
export PYTHONPATH=${REPO}/DiffSynth-Studio

# ---- dataset and checkpoints --------------------------------------------------------
PROMPTS=${PROMPTS:-${ROOT}/Aesthetic-Train-V2/metadata_test.csv}
DATASET_DIR=${DATASET_DIR:-${ROOT}/Aesthetic-Train-V2}
NOFOV_LORA=${NOFOV_LORA:-${ROOT}/foveated_diffusion/experiments/finetuned_no_fov/step-15000.safetensors}
FOV_LORA=${FOV_LORA:-${ROOT}/foveated_diffusion/experiments/rank_32_fov_more_steps/step-5000.safetensors}

# ---- outputs ------------------------------------------------------------------------
OUT_ROOT=${OUT_ROOT:-${ROOT}/foveated_diffusion/results/eval_v2}
LOG_DIR=${LOG_DIR:-${ROOT}/foveated_diffusion/logs/eval_v2}

# ---- sweep definition ---------------------------------------------------------------
MASK_CONFIGS=${MASK_CONFIGS:-"square:0.1 square:0.25 square:0.5 circular:0.25 circular:0.75"}
METHODS=${METHODS:-"high_res naive ours tome ralu"}
NUM_SHARDS=${NUM_SHARDS:-8}          # array size in shard mode; shards per job in node mode
NUM_PROMPTS=${NUM_PROMPTS:-}         # empty = all 10,000

RANDOM_MASK_CENTER=${RANDOM_MASK_CENTER:-false}
MASK_CENTER_SEED=${MASK_CENTER_SEED:-42}
MASK_CENTER_RANGE=${MASK_CENTER_RANGE:-0.3}
CLAMP_MASK_TO_FRAME=${CLAMP_MASK_TO_FRAME:-true}

HEIGHT=${HEIGHT:-1024}
WIDTH=${WIDTH:-1024}
STEPS=${STEPS:-50}
SEED=${SEED:-42}
CFG=${CFG:-4.0}
LR_FACTOR=${LR_FACTOR:-2}
MERGE_MLP=${MERGE_MLP:-true}
RALU_STAGES=${RALU_STAGES:-3}
RALU_LEVEL=${RALU_LEVEL:-4}
RALU_Z_MAX=${RALU_Z_MAX:-6.5}

# ---- Slurm --------------------------------------------------------------------------
# preempt keeps interruptible generation off the batch quota (marlowe_launch_rules.md).
# example.sh pairs preempt with marlowe-m000051 and batch with marlowe-m000051-pm05; this
# sweep charges marlowe-m000051. Override ACCOUNT if you move it to batch.
PARTITION=${PARTITION:-preempt}
ACCOUNT=${ACCOUNT:-marlowe-m000051}

# Per-element resources. One GPU per shard, with the proportional share of a node that
# example.sh allocates whole (8 GPUs / 64 CPUs / 1024 GB).
GEN_TIME=${GEN_TIME:-01:00:00}       # rules doc: one-hour preempt allocation + requeue
GEN_GPUS=${GEN_GPUS:-1}
GEN_CPUS=${GEN_CPUS:-8}
GEN_MEM=${GEN_MEM:-128G}

# Whole-node mode (submit-node): one job runs every shard of a config in parallel.
NODE_TIME=${NODE_TIME:-${GEN_TIME}}
NODE_GPUS=${NODE_GPUS:-8}
NODE_CPUS=${NODE_CPUS:-64}
NODE_MEM=${NODE_MEM:-1024G}
# Device ids the node worker pins its shards to; default 0..NUM_SHARDS-1, which is correct
# when the job owns the node. Set explicitly to test on a shared machine.
NODE_GPU_IDS=${NODE_GPU_IDS:-}

# No mail: a full sweep is hundreds of elements. Monitor with ./submit_eval.sh status.
MAIL_TYPE=${MAIL_TYPE:-}
MAIL_USER=${MAIL_USER:-}

# ---- helpers ------------------------------------------------------------------------
activate_python() {
    if [ -n "${CONDA_ENV}" ] && [ -f "${CONDA_SH}" ]; then
        # shellcheck disable=SC1090
        source "${CONDA_SH}"
        conda activate "${CONDA_ENV}" || { echo "conda activate ${CONDA_ENV} failed" >&2; return 1; }
    fi
    PY=${PY:-$(command -v python)}
    [ -x "${PY}" ] || { echo "no python interpreter found (PY=${PY})" >&2; return 1; }
    echo "python: ${PY}"
}

# Node-local, disposable caches. SLURM_TMPDIR when the scheduler provides it, as example.sh
# does; the MPS directories are example.sh's workaround for the MPS bug seen on this cluster.
setup_node_scratch() {
    local base="${SLURM_TMPDIR:-/tmp/${USER}}"
    export TRITON_CACHE_DIR="${base}/triton"
    export TORCHINDUCTOR_CACHE_DIR="${base}/torchinductor"
    export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID:-$$}"
    export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-${SLURM_JOB_ID:-$$}"
    mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
             "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
}

# Everything eval.sh reads from the environment, in one place, so the workers and the
# launcher cannot disagree about the configuration. Exported rather than assembled into an
# `env VAR=... ` argument list, because MASK_CONFIGS and METHODS contain spaces.
export_eval_env() {
    export PROMPTS NUM_PROMPTS OUT_ROOT LOG_DIR MASK_CONFIGS METHODS \
           HEIGHT WIDTH STEPS SEED CFG LR_FACTOR MERGE_MLP \
           RALU_STAGES RALU_LEVEL RALU_Z_MAX \
           RANDOM_MASK_CENTER MASK_CENTER_SEED MASK_CENTER_RANGE CLAMP_MASK_TO_FRAME \
           NOFOV_LORA FOV_LORA PY \
           DIFFSYNTH_MODEL_BASE_PATH DIFFSYNTH_SKIP_DOWNLOAD PYTHONPATH
}

# Output directory for one (method, config) — must agree with eval.sh's own layout.
eval_out_dir() {   # eval_out_dir <method> <shape:radius>
    if [ "$1" = "high_res" ]; then
        echo "${OUT_ROOT}/high_res"
    else
        local tag="${2%%:*}_r${2##*:}"
        [ "${RANDOM_MASK_CENTER}" = "true" ] && tag="${tag}_randcenter${MASK_CENTER_SEED}"
        echo "${OUT_ROOT}/${tag}/$1"
    fi
}
