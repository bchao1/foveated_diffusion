#!/bin/bash
# =====================================================================================
# ICLR 2027 rebuttal: architecture-ablation LoRA training on Marlowe, cycle pm07.
# Sourced by submit_ablation.sh and train_ablation_worker.sh. Every path is one variable.
#
# The two ablations copy the reference run and change exactly one DiT flag each. Reference:
# wandb flux2-training/cbx1ng9w "rank_32_fov" -> eccv2026_final/rank_32_fov/step-10000,
# commit ce1c906, trained on apollo with 6 x A6000 (effective batch 6), 10k steps, 6.1 h.
# Its flags equal no_tokenAE_lora.sh plus --max_training_steps 10000.
#   abl_rope_standard   --rope_mode standard       (no phase-aligned RoPE)
#   abl_res_embed       --resolution_embedding     (phase-aligned RoPE + HR/LR embedding)
# =====================================================================================

# ---- cycle and storage (Marlowe email, cycle starting 2026-10-30) ---------------------
CYCLE_ACCOUNT=${CYCLE_ACCOUNT:-marlowe-m000051-pm07}   # paid batch partition
BASIC_ACCOUNT=${BASIC_ACCOUNT:-marlowe-m000051}        # preempt partition
CYCLE_SCRATCH=${CYCLE_SCRATCH:-/scratch/m000051-pm07/brianchc}   # purged after the cycle
BASE_SCRATCH=${BASE_SCRATCH:-/scratch/m000051/brian}             # survives cycles

# A git worktree pinned to the ablation commit, separate from the main Marlowe checkout
# (/users/brianchc/foveated_diffusion has local commits and edits; leave it alone).
REPO=${REPO:-/users/brianchc/projects/foveated_diffusion_iclr_abl}
DATASET_DIR=${DATASET_DIR:-${CYCLE_SCRATCH}/datasets/Aesthetic-Train-V2}
TRAIN_CSV=${TRAIN_CSV:-${DATASET_DIR}/metadata_train.csv}
EXP_ROOT=${EXP_ROOT:-${CYCLE_SCRATCH}/foveated_diffusion/experiments}
LOG_ROOT=${LOG_ROOT:-${CYCLE_SCRATCH}/foveated_diffusion/logs}

CONDA_SH=${CONDA_SH:-/cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-${BASE_SCRATCH}/conda_environments/camera_ctrl}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-${BASE_SCRATCH}/models}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-True}
export PYTHONNOUSERSITE=1
export PYTHONPATH=${REPO}/DiffSynth-Studio

# ---- resources ----------------------------------------------------------------------
TRAIN_GPUS=${TRAIN_GPUS:-8}           # batch partition: whole node
# Processes = effective batch. The reference used 6 (apollo); the user chose 8 per arm
# (2026-11-06), which matches the paper's stated batch of 8. Set TRAIN_PROCS=6 to copy it.
TRAIN_PROCS=${TRAIN_PROCS:-8}
TRAIN_TIME=${TRAIN_TIME:-12:00:00}     # reference: ~36 min / 1k steps on H100 -> ~6 h
SMOKE_TIME=${SMOKE_TIME:-01:00:00}
EVAL_TIME=${EVAL_TIME:-12:00:00}       # 10k images on 8 GPUs

# ---- evaluation: the Table 1 setting (iclr_2027_base/circular_r0.25_randcenter42) ------
RESULTS_ROOT=${RESULTS_ROOT:-${CYCLE_SCRATCH}/foveated_diffusion/results/iclr_abl}
EVAL_PROMPTS=${EVAL_PROMPTS:-${DATASET_DIR}/metadata_test.csv}
EVAL_MASK=${EVAL_MASK:-circular:0.25}
REF_LORA=${REF_LORA:-${EXP_ROOT}/reference/rank_32_fov/step-10000.safetensors}
# eval arm -> LoRA checkpoint
eval_lora() {
    case "$1" in
        ref_rank32fov_10k) echo "${REF_LORA}" ;;
        abl_rope_standard|abl_res_embed) echo "${EXP_ROOT}/$1/step-${MAX_STEPS}.safetensors" ;;
        *) echo "unknown eval arm: $1" >&2; return 1 ;;
    esac
}

# ---- the reference recipe (marlowe_scripts/no_tokenAE_lora.sh), verbatim -------------
MAX_STEPS=${MAX_STEPS:-10000}
VAL_PROMPT="Golden retriever on a grassy meadow. Thick sun-kissed fur, clean and glossy, subtle lighter ruff around the neck. Alert amber eyes, black moist nose, relaxed open mouth with pink tongue. Perky ears tilted forward, sturdy athletic stance, tail held in a gentle arc. Warm late-afternoon backdrop, rolling wild grass bending in the breeze, blurred tree line and distant hills under a pastel sky."
LORA_TARGETS="to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj"
for i in $(seq 0 19); do LORA_TARGETS="${LORA_TARGETS},single_transformer_blocks.${i}.attn.to_out"; done
MODEL_FILES="black-forest-labs/FLUX.2-klein-base-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:vae/diffusion_pytorch_model.safetensors"

# train_args <run> <output_dir>: prints one argument per line (read with mapfile).
train_args() {
    local run="$1" out="$2"
    local common=(
        --dataset_base_path "${DATASET_DIR}"
        --dataset_metadata_path "${TRAIN_CSV}"
        --max_pixels 1048576 --height 1024 --width 1024 --dataset_repeat 1
        --model_id_with_origin_paths "${MODEL_FILES}"
        --tokenizer_path "black-forest-labs/FLUX.2-klein-base-4B:tokenizer/"
        --learning_rate 1e-4 --tokenAE_learning_rate 1e-4 --tokenAE_train_steps 0
        --num_epochs 1
        --remove_prefix_in_ckpt "pipe.dit."
        --output_path "${out}"
        --lora_base_model "dit" --lora_target_modules "${LORA_TARGETS}" --lora_rank 32
        --use_gradient_checkpointing
        --dataset_num_workers 8
        --task sft_no_tokenAE
        --find_unused_parameters
        --decode_mode "merge"
    )
    local extra=()
    case "$run" in
        abl_rope_standard)       extra=(--rope_mode standard) ;;
        abl_res_embed)           extra=(--resolution_embedding) ;;
        smoke_abl_rope_standard) extra=(--rope_mode standard) ;;
        smoke_abl_res_embed)     extra=(--resolution_embedding) ;;
        *) echo "unknown run: $run" >&2; return 1 ;;
    esac
    if [[ "$run" == smoke_* ]]; then
        # Load/forward/backward/save path only: no wandb, no validation images.
        common+=(--max_training_steps 20 --save_steps 10 --validation_steps 1000000)
    else
        common+=(--max_training_steps "${MAX_STEPS}" --save_steps 1000
                 --use_wandb --wandb_project "flux2-training" --wandb_run_name "${run}"
                 --validation_prompts "${VAL_PROMPT}" --validation_steps 100
                 --validation_height 1024 --validation_width 1024 --num_validation_images 1)
    fi
    printf '%s\n' "${common[@]}" "${extra[@]}"
}

activate_python() {
    # shellcheck disable=SC1090
    source "${CONDA_SH}" && conda activate "${CONDA_ENV}" || { echo "conda activate ${CONDA_ENV} failed" >&2; return 1; }
    PY=$(command -v python)
    echo "python: ${PY}"
    "${PY}" -c "import torch, diffsynth, sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count()); print('diffsynth', diffsynth.__file__)"
}

setup_node_scratch() {
    local base="${SLURM_TMPDIR:-/tmp/${USER}}/${SLURM_JOB_ID:-$$}"
    export TRITON_CACHE_DIR="${base}/triton"
    export TORCHINDUCTOR_CACHE_DIR="${base}/torchinductor"
    export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID:-$$}"
    export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-${SLURM_JOB_ID:-$$}"
    mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
}
