#!/bin/bash
# =====================================================================================
# ToMe-2x2 (importance token merging) baseline — large-scale eval over mask configs.
#
# The ToMe counterpart of eval/naive.sh: same prompt CSV, same seed, same LoRA, and the
# mask taken straight from eval/inference.py:create_foveation_mask (imported read-only by
# tome_fov_inference.py), sharded over GPUs, writing img_%010d.png + metadata_%05d.csv so
# the existing metric scripts work unchanged.
#
# Supersedes tome_eval.sh, which is a single-config 1k-prompt run on claude_prompts.csv.
#
#   ./tome_eval_10k.sh            # plan, then run every config in MASK_CONFIGS
#   ./tome_eval_10k.sh plan       # print configs, dirs, token budgets — no GPU work
#   ./tome_eval_10k.sh run        # run without the plan preamble
#   ./tome_eval_10k.sh check      # count PNGs in each output dir against the prompt count
#
# Anything in the CONFIGURE block below can also be overridden inline:
#   GPUS="1 7 8" ./tome_eval_10k.sh
#   MASK_CONFIGS="square:0.1" NUM_PROMPTS=100 ./tome_eval_10k.sh
#
# See RUN.md for the knob table and TOME_FOVEATED.md for the port details.
# =====================================================================================
set -uo pipefail

cd "$(dirname "$0")"

MODE=${1:-all}

# =====================================================================================
# CONFIGURE — edit here, or override any line inline as an environment variable.
# =====================================================================================

# ---- mask configs to run -------------------------------------------------------------
# Space-separated "shape:radius" pairs. shape is `square` or `circular`; radius has the
# same semantics as eval/inference.py --mask_radius (square: side as a fraction of the
# image; circular: radius relative to half the diagonal).
#
# The default is the five configs that have completed naive_mixed_res_* result dirs, so
# each ToMe dir has a naive/ours dir at the identical mask to compare against:
#     square:0.1     <-> naive_mixed_res_r_0.1_eval
#     square:0.25    <-> naive_mixed_res_r_0.25_eval
#     square:0.5     <-> naive_mixed_res_eval
#     circular:0.25  <-> naive_mixed_res_circle_r_0.25_eval
#     circular:0.75  <-> naive_mixed_res_circle_r_0.75_eval
#
# Add/remove pairs freely, e.g.  MASK_CONFIGS="square:0.1 circular:0.3"
MASK_CONFIGS=${MASK_CONFIGS:-"square:0.1 square:0.25 square:0.5 circular:0.25 circular:0.75"}

# ---- GPUs ----------------------------------------------------------------------------
# One shard per GPU; --num_subsets is derived from the list, so any count works and the
# prompt set is still covered exactly once (shard i takes global_idx % n == i).
# Check `nvidia-smi` first — these are 10k-image runs and will sit on the GPU for hours.
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}

# ---- prompts -------------------------------------------------------------------------
# Same CSV eval/naive.sh uses: 10,000 rows with an `image,prompt` header.
# Leave NUM_PROMPTS empty for the full set; set it to a small number for a smoke test.
PROMPTS=${PROMPTS:-/local/brian/Aesthetic-Train-V2/metadata_test.csv}
NUM_PROMPTS=${NUM_PROMPTS:-}

# ---- output --------------------------------------------------------------------------
# Directory naming follows the naive convention so the two are easy to pair up:
#   square   -> ${OUT_ROOT}/${OUT_PREFIX}_r_<radius>_eval
#   circular -> ${OUT_ROOT}/${OUT_PREFIX}_circle_r_<radius>_eval
OUT_ROOT=${OUT_ROOT:-/local/brian/foveated_diffusion/results}
OUT_PREFIX=${OUT_PREFIX:-tome_2x2}
LOG_DIR=${LOG_DIR:-${OUT_ROOT}/logs_tome_eval}

# ---- model ---------------------------------------------------------------------------
# ToMe is training-free, so it runs on the same no-foveation LoRA as eval/naive.sh.
# Set LORA="" to run the base model with no LoRA at all.
# MODEL_PATH points at a local FLUX.2 directory and overrides MODEL_ID when non-empty.
step=${step:-15000}
ckpt_folder=${ckpt_folder:-/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/}
LORA=${LORA:-${ckpt_folder}/step-${step}.safetensors}
MODEL_ID=${MODEL_ID:-black-forest-labs/FLUX.2-klein-base-4B}
MODEL_PATH=${MODEL_PATH:-}

# ---- generation (must match eval/naive.sh) -------------------------------------------
HEIGHT=${HEIGHT:-1024}
WIDTH=${WIDTH:-1024}
STEPS=${STEPS:-50}                  # inference.py's default, which naive.sh leaves alone
SEED=${SEED:-42}
CFG=${CFG:-4.0}                     # true CFG; naive.sh runs inference.py's default 4.0
LR_FACTOR=${LR_FACTOR:-2}           # periphery downsample factor (2 -> 2x2 merged blocks)

# ---- randomized mask center ----------------------------------------------------------
# false = mask centered (default, matches eval/naive.sh today).
# true  = prompt i gets its own fixation point from inference.py:foveation_center_for_index,
#         drawn uniformly from [-RANGE, RANGE]^2. Index-addressable, so it does not depend
#         on GPU count, and identical across ours/naive/ToMe/RALU given the same
#         MASK_CENTER_SEED and MASK_CENTER_RANGE. Pass the SAME values to every baseline.
# Centers land in metadata_*.csv as center_x/center_y. OUT_SUFFIX keeps randomized runs in
# their own dirs so they never mix with the centered ones.
RANDOM_MASK_CENTER=${RANDOM_MASK_CENTER:-false}
MASK_CENTER_SEED=${MASK_CENTER_SEED:-42}
MASK_CENTER_RANGE=${MASK_CENTER_RANGE:-0.3}
if [ "${RANDOM_MASK_CENTER}" = "true" ]; then
    OUT_SUFFIX=${OUT_SUFFIX:-_randcenter${MASK_CENTER_SEED}}
else
    OUT_SUFFIX=${OUT_SUFFIX:-}
fi

# ---- token merging -------------------------------------------------------------------
# MERGE_MLP=true is the headline config: it also merges the image MLP in the 8 double
# blocks, which puts ToMe within 0.9% of our wall-clock at the same token budget, i.e. a
# quality-at-equal-cost comparison. MERGE_MLP=false only handicaps it (see RUN.md).
# ENABLE_TOME=false is the identity control and should reproduce high_res.
MERGE_MLP=${MERGE_MLP:-true}
ENABLE_TOME=${ENABLE_TOME:-true}

# ---- behaviour -----------------------------------------------------------------------
# SKIP_COMPLETE=1 skips a config whose output dir already holds the expected PNG count,
# so an interrupted sweep can be resumed by simply re-running this script.
SKIP_COMPLETE=${SKIP_COMPLETE:-1}

# =====================================================================================
# END CONFIGURE — nothing below needs editing for a normal run.
# =====================================================================================

REPO=${REPO:-/home/brianchc/foveated_diffusion-private}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/miele/brian/modelscope}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-True}
# imp_tome otherwise resolves `diffsynth` to /home/brianchc/DiffSynth-Studio, a different
# checkout without the foveated pipeline. Force this repo's copy.
export PYTHONPATH=${PYTHONPATH:-${REPO}/DiffSynth-Studio}

PY=${PY:-/home/brianchc/miniconda3/envs/imp_tome/bin/python}
SCRIPT=${SCRIPT:-tome_fov_inference.py}

gpu_arr=($GPUS)
NUM_SUBSETS=${#gpu_arr[@]}

model_args=()
if [ -n "${MODEL_PATH}" ]; then model_args=(--model_path "${MODEL_PATH}")
else                            model_args=(--model_id "${MODEL_ID}"); fi

# Number of images a complete run must produce: the prompt count, capped by NUM_PROMPTS.
EXPECTED=$($PY -c '
import sys, pandas as pd
n = len(pd.read_csv(sys.argv[1]))
cap = sys.argv[2].strip()
print(min(n, int(cap)) if cap else n)
' "$PROMPTS" "${NUM_PROMPTS}") || { echo "cannot read $PROMPTS" >&2; exit 1; }

out_dir() {   # out_dir <shape> <radius>
    local shape="$1" radius="$2"
    if [ "$shape" = "circular" ]; then
        echo "${OUT_ROOT}/${OUT_PREFIX}_circle_r_${radius}${OUT_SUFFIX}_eval"
    else
        echo "${OUT_ROOT}/${OUT_PREFIX}_r_${radius}${OUT_SUFFIX}_eval"
    fi
}

png_count() { ls "$1"/img_*.png 2>/dev/null | wc -l; }

run_subset() {   # run_subset <gpu_id> <subset_idx> <shape> <radius>
    local gpu_id="$1" subset_idx="$2" shape="$3" radius="$4"
    local dir; dir=$(out_dir "$shape" "$radius")
    local cap=() lora=()
    [ -n "${NUM_PROMPTS}" ] && cap=(--num_prompts "${NUM_PROMPTS}")
    [ -n "${LORA}" ] && lora=(--lora_checkpoint "${LORA}")

    CUDA_VISIBLE_DEVICES=$gpu_id $PY $SCRIPT \
        "${model_args[@]}" "${lora[@]}" \
        --output_dir "$dir" \
        --full_eval \
        --prompt_dataset_path "${PROMPTS}" \
        "${cap[@]}" \
        --height "${HEIGHT}" --width "${WIDTH}" \
        --full_eval_mask "${shape}" \
        --mask_radius "${radius}" \
        --lr_downsample_factor "${LR_FACTOR}" \
        --random_mask_center "${RANDOM_MASK_CENTER}" \
        --mask_center_seed "${MASK_CENTER_SEED}" \
        --mask_center_range "${MASK_CENTER_RANGE}" \
        --merge_mlp "${MERGE_MLP}" \
        --enable_tome "${ENABLE_TOME}" \
        --num_inference_steps "${STEPS}" \
        --guidance_scale "${CFG}" \
        --seed "${SEED}" \
        --num_subsets "${NUM_SUBSETS}" \
        --subset_idx "${subset_idx}" \
        > "${LOG_DIR}/$(basename "$dir")_shard${subset_idx}.log" 2>&1
}

run_config() {   # run_config <shape> <radius>
    local shape="$1" radius="$2"
    local dir; dir=$(out_dir "$shape" "$radius")
    local have; have=$(png_count "$dir")

    if [ "${SKIP_COMPLETE}" = "1" ] && [ "$have" -ge "$EXPECTED" ]; then
        echo "== skip  ${shape} r=${radius}  (already ${have}/${EXPECTED} pngs in ${dir})"
        return 0
    fi
    mkdir -p "$dir" "$LOG_DIR"
    echo "== run   ${shape} r=${radius}  ->  ${dir}"
    echo "   ${NUM_SUBSETS} shards on GPUs ${GPUS}; logs in ${LOG_DIR}; have ${have}/${EXPECTED}"

    local i=0
    for g in ${GPUS}; do
        run_subset "$g" "$i" "$shape" "$radius" &
        i=$((i + 1))
    done
    wait

    have=$(png_count "$dir")
    if [ "$have" -eq "$EXPECTED" ]; then
        echo "   OK    ${have}/${EXPECTED} pngs"
    else
        echo "   WARN  ${have}/${EXPECTED} pngs — check ${LOG_DIR}/$(basename "$dir")_shard*.log" >&2
    fi
}

print_plan() {
    echo "======================================================================"
    echo " ToMe-2x2 large-scale eval"
    echo "   prompts       ${PROMPTS}"
    echo "   images/config ${EXPECTED}${NUM_PROMPTS:+ (capped at ${NUM_PROMPTS})}"
    echo "   configs       ${MASK_CONFIGS}"
    echo "   gpus          ${GPUS}  (${NUM_SUBSETS} shards)"
    echo "   tome          merge_mlp=${MERGE_MLP} enable_tome=${ENABLE_TOME}, ${STEPS} steps"
    echo "   mask          lr_factor=${LR_FACTOR} at ${HEIGHT}x${WIDTH}, from inference.py:create_foveation_mask"
    if [ "${RANDOM_MASK_CENTER}" = "true" ]; then
        echo "   mask center   RANDOM per prompt (seed ${MASK_CENTER_SEED}, range +/-${MASK_CENTER_RANGE})"
        echo "                 pass the same seed/range to ours, naive and RALU"
    else
        echo "   mask center   centered"
    fi
    echo "   model         ${MODEL_PATH:-${MODEL_ID}}"
    echo "   lora          ${LORA:-<none>}"
    echo "   seed/cfg      ${SEED} / ${CFG}"
    echo "   total images  $((EXPECTED * $(echo "$MASK_CONFIGS" | wc -w)))"
    echo "----------------------------------------------------------------------"
    # Token budget per config, from the same mask function the run itself will call.
    $PY - <<EOF
import sys
sys.path.insert(0, "..")
from inference import create_foveation_mask
for cfg in "${MASK_CONFIGS}".split():
    shape, r = cfg.split(":")
    m = create_foveation_mask(${HEIGHT}, ${WIDTH}, center=(0, 0), r=float(r),
                              shape=shape, lr_factor=${LR_FACTOR})
    hr, tot = int(m.sum()), m.numel()
    kept = hr + (tot - hr) // (${LR_FACTOR} ** 2)
    print(f"   {shape:9s} r={r:<5s} hr_tokens={hr:5d}/{tot}  "
          f"merged {tot} -> {kept} ({100*kept/tot:.1f}%)")
EOF
    echo "----------------------------------------------------------------------"
    for cfg in ${MASK_CONFIGS}; do
        local shape="${cfg%%:*}" radius="${cfg##*:}"
        local dir; dir=$(out_dir "$shape" "$radius")
        printf "   %-9s r=%-5s %-6s pngs  %s\n" "$shape" "$radius" "$(png_count "$dir")" "$dir"
    done
    echo "======================================================================"
}

case "${MODE}" in
  plan)
    print_plan
    ;;
  check)
    for cfg in ${MASK_CONFIGS}; do
        shape="${cfg%%:*}"; radius="${cfg##*:}"
        dir=$(out_dir "$shape" "$radius")
        have=$(png_count "$dir")
        meta=$(ls "$dir"/metadata_*.csv 2>/dev/null | wc -l)
        status="INCOMPLETE"; [ "$have" -eq "$EXPECTED" ] && status="COMPLETE"
        printf "%-10s %-11s %6s/%-6s pngs  %s shard csv  %s\n" \
               "$status" "${shape}:${radius}" "$have" "$EXPECTED" "$meta" "$dir"
    done
    ;;
  all|run)
    [ "${MODE}" = "all" ] && print_plan
    mkdir -p "$LOG_DIR"
    for cfg in ${MASK_CONFIGS}; do
        run_config "${cfg%%:*}" "${cfg##*:}"
    done
    echo "all configs done."
    ;;
  *)
    echo "usage: $0 {plan|run|check|all}" >&2
    exit 1
    ;;
esac
