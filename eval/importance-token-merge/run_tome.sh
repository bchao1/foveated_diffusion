#!/bin/bash
# Parameterized launcher for the ToMe-2x2 foveated baseline.
#
#   ./run_tome.sh single     one prompt, one GPU               (default)
#   ./run_tome.sh eval       full prompt set, sharded over GPUS
#   ./run_tome.sh sweep      one prompt at several mask radii
#   ./run_tome.sh compare    ToMe vs merging-off vs ours, one prompt each
#
# Every knob is an environment variable with a default. Override inline:
#
#   STEPS=28 RADIUS=0.5 MERGE_MLP=true ./run_tome.sh single
#   MODEL_PATH=/local/models/FLUX.2-klein-base-4B ./run_tome.sh single
#   GPUS="0 1 2 3" NUM_PROMPTS=500 ./run_tome.sh eval
#
# See RUN.md for the full knob table.
set -euo pipefail

MODE=${1:-single}

# ---------------------------------------------------------------- model
MODEL_ID=${MODEL_ID:-black-forest-labs/FLUX.2-klein-base-4B}
MODEL_PATH=${MODEL_PATH:-}                 # local dir; overrides MODEL_ID when set
LORA=${LORA:-/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/step-15000.safetensors}
FOV_LORA=${FOV_LORA:-/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-5000.safetensors}

# ------------------------------------------------------------ generation
STEPS=${STEPS:-50}
SEED=${SEED:-42}
CFG=${CFG:-4.0}
HEIGHT=${HEIGHT:-1024}
WIDTH=${WIDTH:-1024}

# -------------------------------------------------------- foveation mask
MASK=${MASK:-circular}                     # circular | square | checkerboard
RADIUS=${RADIUS:-0.3}                      # fovea size; "mask ratio"
LR_FACTOR=${LR_FACTOR:-2}                  # periphery downsample factor (2 -> 2x2 blocks)

# --------------------------------------------------------------- token merge
MERGE_MLP=${MERGE_MLP:-false}              # also merge the image MLP in the 8 double blocks
ENABLE_TOME=${ENABLE_TOME:-true}           # false = identity control

# ---------------------------------------------------------------- runtime
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
OUT=${OUT:-../../results/tome}
PROMPT=${PROMPT:-"A lion lounges calmly, displaying its powerful physique, with a soft focus highlighting its majestic mane and intense gaze."}
PROMPTS_CSV=${PROMPTS_CSV:-../prompts/claude_prompts.csv}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
RADII=${RADII:-"0.1 0.2 0.3 0.4 0.5"}      # sweep mode only

PY=${PY:-/home/brianchc/miniconda3/envs/imp_tome/bin/python}
REPO=${REPO:-/home/brianchc/foveated_diffusion-private}

export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/miele/brian/modelscope}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-True}
# imp_tome otherwise resolves `diffsynth` to /home/brianchc/DiffSynth-Studio, a different checkout.
export PYTHONPATH=${REPO}/DiffSynth-Studio

cd "$(dirname "$0")"

model_args=()
if [ -n "${MODEL_PATH}" ]; then model_args=(--model_path "${MODEL_PATH}")
else                            model_args=(--model_id "${MODEL_ID}"); fi

lora_args=()
if [ -n "${LORA}" ]; then lora_args=(--lora_checkpoint "${LORA}"); fi

common=(
    "${model_args[@]}" "${lora_args[@]}"
    --height "${HEIGHT}" --width "${WIDTH}"
    --num_inference_steps "${STEPS}" --guidance_scale "${CFG}" --seed "${SEED}"
    --full_eval_mask "${MASK}" --mask_radius "${RADIUS}" --lr_downsample_factor "${LR_FACTOR}"
    --merge_mlp "${MERGE_MLP}" --enable_tome "${ENABLE_TOME}"
)

first_gpu() { echo "${GPUS}" | awk '{print $1}'; }

banner() {
    echo "=============================================================="
    echo " mode=${MODE}  steps=${STEPS}  mask=${MASK} r=${RADIUS} lr_factor=${LR_FACTOR}"
    echo " merge_mlp=${MERGE_MLP}  enable_tome=${ENABLE_TOME}  seed=${SEED}"
    echo " model=${MODEL_PATH:-${MODEL_ID}}"
    echo " out=${OUT}"
    echo "=============================================================="
}

case "${MODE}" in

single)
    banner
    CUDA_VISIBLE_DEVICES=$(first_gpu) $PY tome_fov_inference.py \
        "${common[@]}" --prompt "${PROMPT}" --output_dir "${OUT}/single"
    echo "-> ${OUT}/single"
    ;;

eval)
    banner
    n=$(echo "${GPUS}" | wc -w)
    echo " sharding ${NUM_PROMPTS} prompts over ${n} GPUs (${GPUS})"
    i=0
    for g in ${GPUS}; do
        CUDA_VISIBLE_DEVICES=$g $PY tome_fov_inference.py \
            "${common[@]}" \
            --full_eval --prompt_dataset_path "${PROMPTS_CSV}" --num_prompts "${NUM_PROMPTS}" \
            --num_subsets "${n}" --subset_idx "${i}" \
            --output_dir "${OUT}/eval" &
        i=$((i + 1))
    done
    wait
    echo "-> ${OUT}/eval  (img_%010d.png + metadata_%05d.csv)"
    ;;

sweep)
    banner
    echo " radii: ${RADII}"
    i=0
    gpu_arr=(${GPUS})
    for r in ${RADII}; do
        g=${gpu_arr[$((i % ${#gpu_arr[@]}))]}
        CUDA_VISIBLE_DEVICES=$g $PY tome_fov_inference.py \
            "${model_args[@]}" "${lora_args[@]}" \
            --height "${HEIGHT}" --width "${WIDTH}" \
            --num_inference_steps "${STEPS}" --guidance_scale "${CFG}" --seed "${SEED}" \
            --full_eval_mask "${MASK}" --mask_radius "$r" --lr_downsample_factor "${LR_FACTOR}" \
            --merge_mlp "${MERGE_MLP}" \
            --prompt "${PROMPT}" --output_dir "${OUT}/sweep/r${r}" &
        i=$((i + 1))
    done
    wait
    echo "-> ${OUT}/sweep/r*"
    ;;

compare)
    banner
    gpu_arr=(${GPUS})
    g0=${gpu_arr[0]}; g1=${gpu_arr[1 % ${#gpu_arr[@]}]}
    g2=${gpu_arr[2 % ${#gpu_arr[@]}]}; g3=${gpu_arr[3 % ${#gpu_arr[@]}]}

    # ToMe-2x2, attention-only in the double blocks
    CUDA_VISIBLE_DEVICES=$g0 $PY tome_fov_inference.py \
        "${common[@]}" --merge_mlp false --prompt "${PROMPT}" \
        --output_dir "${OUT}/compare/tome_attn" &
    # ToMe-2x2, image MLP merged in the double blocks too
    CUDA_VISIBLE_DEVICES=$g1 $PY tome_fov_inference.py \
        "${common[@]}" --merge_mlp true --prompt "${PROMPT}" \
        --output_dir "${OUT}/compare/tome_attn_mlp" &
    # Identity control: same code path, merging disabled
    CUDA_VISIBLE_DEVICES=$g2 $PY tome_fov_inference.py \
        "${common[@]}" --enable_tome false --prompt "${PROMPT}" \
        --output_dir "${OUT}/compare/tome_off" &
    # Ours, same mask and step count
    CUDA_VISIBLE_DEVICES=$g3 $PY ../inference.py \
        --experiment ours --decode_mode merge --tokenAE_training false \
        --lora_checkpoint "${FOV_LORA}" \
        --height "${HEIGHT}" --width "${WIDTH}" \
        --num_inference_steps "${STEPS}" --guidance_scale "${CFG}" --seed "${SEED}" \
        --full_eval_mask "${MASK}" --mask_radius "${RADIUS}" --lr_downsample_factor "${LR_FACTOR}" \
        --soft_foveation_blend true --foveation_outline false \
        --prompt "${PROMPT}" --output_dir "${OUT}/compare/ours" &
    wait
    echo "-> ${OUT}/compare/{tome_attn,tome_attn_mlp,tome_off,ours}"
    ;;

*)
    echo "unknown mode: ${MODE}" >&2
    echo "usage: $0 [single|eval|sweep|compare]" >&2
    exit 1
    ;;
esac
