#!/bin/bash
# =====================================================================================
# eval.sh — the single entry point for large-scale evaluation of every arm.
#
# Five methods, one configuration block. Each method function below receives the SAME
# mask flags, the SAME prompt set, the SAME seed/steps/cfg/resolution and the SAME
# sharding; only the method and its model differ:
#
#   high_res   full-resolution reference           inference.py --experiment high_res
#   naive      naive mixed-resolution              inference.py --experiment naive_mixed_res
#   ours       foveated diffusion                  inference.py --experiment ours
#   tome       ToMe-2x2 importance token merging    importance-token-merge/tome_fov_inference.py
#   ralu       RALU 3-stage, compute-matched        RALU/RALU_fov_inference_diffsynth.py
#
# Usage (runnable from any directory):
#   ./eval.sh plan                 print the resolved config, output dirs, counts — no GPU
#   ./eval.sh masks                verify all arms derive an identical mask — no GPU
#   ./eval.sh <method>             run one method over every MASK_CONFIGS entry
#   ./eval.sh all                  run all five, sequentially
#   ./eval.sh check                PNG counts per method/config vs the prompt count
#   ./eval.sh shard <method> <shape:radius> <shard_id> <num_shards>
#                                  one shard, foreground, on the GPU the caller or the
#                                  scheduler already provided. This is what a Slurm array
#                                  element runs; see marlowe_scripts/MARLOWE_EVAL.md.
#
#   NUM_PROMPTS=2 GPUS="5" ./eval.sh ours          # smoke test
#   MASK_CONFIGS="square:0.1" ./eval.sh all        # one mask config, all arms
#   RANDOM_MASK_CENTER=true ./eval.sh all          # randomized fixation points
#   GPUS="0 1 2 3 4 5 6 7" ./eval.sh naive         # shard over 8 GPUs
#
# Every knob is an environment variable with a default; see the CONFIGURE block.
# Methods write img_%010d.png + metadata_%05d.csv, so the metric scripts
# (compute_clip_score.py, compute_hpsv2_score.py, extract_inception_features.py) consume
# any output directory unchanged.
# =====================================================================================
set -uo pipefail

# Resolve the repo from this script's location, so the script works from any cwd.
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${EVAL_DIR}/.." && pwd)"

MODE=${1:-plan}

# =====================================================================================
# CONFIGURE
# =====================================================================================

# ---- what to run ---------------------------------------------------------------------
# Space-separated "shape:radius" pairs; shape is `square` or `circular`, radius has
# eval/inference.py --mask_radius semantics. Every method runs every entry.
# high_res is mask-independent, so it is generated once and shared by all configs.
MASK_CONFIGS=${MASK_CONFIGS:-"square:0.1 square:0.25 square:0.5 circular:0.25 circular:0.75"}
METHODS=${METHODS:-"high_res naive ours tome ralu"}     # used by `all`, `plan`, `check`

# ---- prompts -------------------------------------------------------------------------
# 10,000-row CSV with an `image,prompt` header. NUM_PROMPTS empty = all of them.
PROMPTS=${PROMPTS:-/local/brian/Aesthetic-Train-V2/metadata_test.csv}
NUM_PROMPTS=${NUM_PROMPTS:-}

# ---- GPUs ----------------------------------------------------------------------------
# One shard per GPU. Shard i takes prompts where (global_idx % n == i), so any GPU count
# covers the prompt set exactly once and image indices stay comparable across methods.
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}

# ---- generation, shared by every method ----------------------------------------------
HEIGHT=${HEIGHT:-1024}
WIDTH=${WIDTH:-1024}
STEPS=${STEPS:-50}
SEED=${SEED:-42}
CFG=${CFG:-4.0}                     # true CFG on both sides (inference.py --guidance_scale)
LR_FACTOR=${LR_FACTOR:-2}           # periphery downsample factor

# ---- foveation mask center -----------------------------------------------------------
# false = centered. true = per-prompt fixation point from
# inference.py:foveation_center_for_index(prompt_idx, MASK_CENTER_SEED, range), which is
# index-addressable, so it is identical across methods and independent of GPU count.
# CLAMP_MASK_TO_FRAME caps the range at 0.5 - mask_half_extent - one low-res cell, so the
# mask never cuts the frame edge; the cap is therefore a function of the radius.
RANDOM_MASK_CENTER=${RANDOM_MASK_CENTER:-false}
MASK_CENTER_SEED=${MASK_CENTER_SEED:-42}
MASK_CENTER_RANGE=${MASK_CENTER_RANGE:-0.3}
CLAMP_MASK_TO_FRAME=${CLAMP_MASK_TO_FRAME:-true}

# ---- models --------------------------------------------------------------------------
# The training-free arms (high_res, naive, tome, ralu) use the no-foveation LoRA; ours
# uses the foveated one. Set either to "" for the base model.
NOFOV_LORA=${NOFOV_LORA:-/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/step-15000.safetensors}
FOV_LORA=${FOV_LORA:-/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-5000.safetensors}
TOKENAE_FOLDER=${TOKENAE_FOLDER:-}          # optional, `ours` only; empty = not passed

# ---- method-specific knobs -----------------------------------------------------------
MERGE_MLP=${MERGE_MLP:-true}        # tome: headline config, matches our cost at equal tokens
RALU_STAGES=${RALU_STAGES:-3}       # ralu: 3 = their working design (2 is known broken)
RALU_LEVEL=${RALU_LEVEL:-4}         # ralu: published operating point for N/e proportions
RALU_Z_MAX=${RALU_Z_MAX:-6.5}       # ralu: authors' re-noising strength
OURS_STEPS=${OURS_STEPS:-${STEPS}}  # ralu: compute-match target = our step count

# ---- output --------------------------------------------------------------------------
# Layout groups the arms per mask config, so a comparison is one directory listing:
#   $OUT_ROOT/<shape>_r<radius>[_randcenter<seed>]/<method>/
#   $OUT_ROOT/high_res/                                       (mask-independent, shared)
OUT_ROOT=${OUT_ROOT:-/local/brian/foveated_diffusion/results/eval_v2}
LOG_DIR=${LOG_DIR:-${OUT_ROOT}/logs}

# SKIP_COMPLETE=1 skips a (method, config) whose directory already holds the expected PNG
# count, so re-running the script resumes an interrupted sweep.
SKIP_COMPLETE=${SKIP_COMPLETE:-1}

# =====================================================================================
# END CONFIGURE
# =====================================================================================

export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/miele/brian/modelscope}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-True}
# The imp_tome env otherwise resolves `diffsynth` to /home/brianchc/DiffSynth-Studio, a
# different checkout without the foveated pipeline. Force this repo's copy.
export PYTHONPATH=${PYTHONPATH:-${REPO}/DiffSynth-Studio}

PY=${PY:-/home/brianchc/miniconda3/envs/imp_tome/bin/python}

gpu_arr=($GPUS)
NUM_SUBSETS=${#gpu_arr[@]}

EXPECTED=$($PY -c '
import sys, pandas as pd
n = len(pd.read_csv(sys.argv[1]))
cap = sys.argv[2].strip()
print(min(n, int(cap)) if cap else n)
' "$PROMPTS" "${NUM_PROMPTS}") || { echo "cannot read prompts: $PROMPTS" >&2; exit 1; }

# Run a command pinned to one GPU, or on whatever the scheduler already gave us when the
# gpu id is empty (Slurm sets CUDA_VISIBLE_DEVICES itself — do not override it there).
gpu_run() {   # gpu_run <gpu_id_or_empty> <cmd> [args...]
    local gpu="$1"; shift
    if [ -n "$gpu" ]; then CUDA_VISIBLE_DEVICES="$gpu" "$@"; else "$@"; fi
}

# ---- flag groups every method shares -------------------------------------------------
# Defined once so no arm can drift from another. `mask_args` is what makes the masks
# identical; `eval_args` is what makes the prompt set and sharding identical.
mask_args() {   # mask_args <shape> <radius>
    echo "--full_eval_mask $1 --mask_radius $2 --lr_downsample_factor ${LR_FACTOR} \
          --random_mask_center ${RANDOM_MASK_CENTER} \
          --mask_center_seed ${MASK_CENTER_SEED} \
          --mask_center_range ${MASK_CENTER_RANGE} \
          --clamp_mask_to_frame ${CLAMP_MASK_TO_FRAME}"
}

eval_args() {   # eval_args <subset_idx>
    local cap=""
    [ -n "${NUM_PROMPTS}" ] && cap="--num_prompts ${NUM_PROMPTS}"
    echo "--full_eval --prompt_dataset_path ${PROMPTS} ${cap} \
          --num_subsets ${NUM_SUBSETS} --subset_idx $1 \
          --height ${HEIGHT} --width ${WIDTH} --seed ${SEED}"
}

# =====================================================================================
# The five methods. Each takes (gpu, subset_idx, shape, radius, out_dir) and differs from
# the others only in the script it calls, its --experiment/method flags, and its model.
# =====================================================================================

eval_high_res() {
    local gpu="$1" subset="$2" shape="$3" radius="$4" out="$5"
    local lora=(); [ -n "${NOFOV_LORA}" ] && lora=(--lora_checkpoint "${NOFOV_LORA}")
    gpu_run "$gpu" $PY "${EVAL_DIR}/inference.py" \
        --experiment high_res --decode_mode direct --tokenAE_training false \
        "${lora[@]}" \
        --num_inference_steps "${STEPS}" --guidance_scale "${CFG}" \
        $(mask_args "$shape" "$radius") $(eval_args "$subset") \
        --output_dir "$out"
}

eval_naive() {
    local gpu="$1" subset="$2" shape="$3" radius="$4" out="$5"
    local lora=(); [ -n "${NOFOV_LORA}" ] && lora=(--lora_checkpoint "${NOFOV_LORA}")
    gpu_run "$gpu" $PY "${EVAL_DIR}/inference.py" \
        --experiment naive_mixed_res --decode_mode merge --tokenAE_training false \
        --soft_foveation_blend true --foveation_outline false \
        "${lora[@]}" \
        --num_inference_steps "${STEPS}" --guidance_scale "${CFG}" \
        $(mask_args "$shape" "$radius") $(eval_args "$subset") \
        --output_dir "$out"
}

eval_ours() {
    local gpu="$1" subset="$2" shape="$3" radius="$4" out="$5"
    local lora=(); [ -n "${FOV_LORA}" ] && lora=(--lora_checkpoint "${FOV_LORA}")
    local tae=(); [ -n "${TOKENAE_FOLDER}" ] && tae=(--tokenAE_folder "${TOKENAE_FOLDER}")
    gpu_run "$gpu" $PY "${EVAL_DIR}/inference.py" \
        --experiment ours --decode_mode merge --tokenAE_training false \
        --soft_foveation_blend true --foveation_outline false \
        "${lora[@]}" "${tae[@]}" \
        --num_inference_steps "${STEPS}" --guidance_scale "${CFG}" \
        $(mask_args "$shape" "$radius") $(eval_args "$subset") \
        --output_dir "$out"
}

eval_tome() {
    local gpu="$1" subset="$2" shape="$3" radius="$4" out="$5"
    local lora=(); [ -n "${NOFOV_LORA}" ] && lora=(--lora_checkpoint "${NOFOV_LORA}")
    gpu_run "$gpu" $PY "${EVAL_DIR}/importance-token-merge/tome_fov_inference.py" \
        --merge_mlp "${MERGE_MLP}" --enable_tome true \
        "${lora[@]}" \
        --num_inference_steps "${STEPS}" --guidance_scale "${CFG}" \
        $(mask_args "$shape" "$radius") $(eval_args "$subset") \
        --output_dir "$out"
}

eval_ralu() {
    local gpu="$1" subset="$2" shape="$3" radius="$4" out="$5"
    local lora=(); [ -n "${NOFOV_LORA}" ] && lora=(--lora_checkpoint "${NOFOV_LORA}")
    # --cfg_scale is RALU's name for the same true-CFG knob inference.py calls
    # --guidance_scale; --num_inference_steps is only a fallback since --match_compute
    # solves the total from OURS_STEPS.
    gpu_run "$gpu" $PY "${EVAL_DIR}/RALU/RALU_fov_inference_diffsynth.py" \
        --num_stages "${RALU_STAGES}" --ralu_level "${RALU_LEVEL}" \
        --match_compute --ours_steps "${OURS_STEPS}" --z_max "${RALU_Z_MAX}" \
        "${lora[@]}" \
        --num_inference_steps "${STEPS}" --cfg_scale "${CFG}" \
        $(mask_args "$shape" "$radius") $(eval_args "$subset") \
        --output_dir "$out"
}

# =====================================================================================
# Driver
# =====================================================================================

mask_tag() {   # mask_tag <shape> <radius>
    local tag="$1_r$2"
    [ "${RANDOM_MASK_CENTER}" = "true" ] && tag="${tag}_randcenter${MASK_CENTER_SEED}"
    echo "$tag"
}

out_dir() {   # out_dir <method> <shape> <radius>
    # high_res ignores the mask, so it is generated once and shared by every config.
    if [ "$1" = "high_res" ]; then echo "${OUT_ROOT}/high_res"
    else echo "${OUT_ROOT}/$(mask_tag "$2" "$3")/$1"; fi
}

png_count() { ls "$1"/img_*.png 2>/dev/null | wc -l; }

is_method() {
    case "$1" in high_res|naive|ours|tome|ralu) return 0 ;; *) return 1 ;; esac
}

run_one() {   # run_one <method> <shape> <radius>
    local method="$1" shape="$2" radius="$3"
    local out; out=$(out_dir "$method" "$shape" "$radius")
    local have; have=$(png_count "$out")

    if [ "${SKIP_COMPLETE}" = "1" ] && [ "$have" -ge "$EXPECTED" ]; then
        echo "== skip  ${method} ${shape} r=${radius}  (${have}/${EXPECTED} pngs present)"
        return 0
    fi
    mkdir -p "$out" "$LOG_DIR"
    echo "== run   ${method} ${shape} r=${radius}  ->  ${out}"
    echo "         ${NUM_SUBSETS} shards on GPUs ${GPUS}, have ${have}/${EXPECTED}"

    local i=0 log_tag
    log_tag="${method}_$(mask_tag "$shape" "$radius")"
    for g in ${GPUS}; do
        "eval_${method}" "$g" "$i" "$shape" "$radius" "$out" \
            > "${LOG_DIR}/${log_tag}_shard${i}.log" 2>&1 &
        i=$((i + 1))
    done
    wait

    have=$(png_count "$out")
    if [ "$have" -eq "$EXPECTED" ]; then
        echo "         OK   ${have}/${EXPECTED} pngs"
    else
        echo "         WARN ${have}/${EXPECTED} pngs — see ${LOG_DIR}/${log_tag}_shard*.log" >&2
    fi
}

run_method() {   # run_method <method>: every mask config (once, for high_res)
    local method="$1"
    if [ "$method" = "high_res" ]; then
        local first; first=$(echo "${MASK_CONFIGS}" | awk '{print $1}')
        run_one high_res "${first%%:*}" "${first##*:}"      # mask flags are ignored
        return
    fi
    for cfg in ${MASK_CONFIGS}; do
        run_one "$method" "${cfg%%:*}" "${cfg##*:}"
    done
}

print_plan() {
    echo "======================================================================"
    echo " eval.sh — large-scale evaluation"
    echo "   repo          ${REPO}"
    echo "   methods       ${METHODS}"
    echo "   mask configs  ${MASK_CONFIGS}"
    echo "   prompts       ${PROMPTS}"
    echo "   images/arm    ${EXPECTED}${NUM_PROMPTS:+ (capped at ${NUM_PROMPTS})}"
    echo "   gpus          ${GPUS}  (${NUM_SUBSETS} shards)"
    echo "   generation    ${HEIGHT}x${WIDTH}, ${STEPS} steps, cfg ${CFG}, seed ${SEED}, lr_factor ${LR_FACTOR}"
    if [ "${RANDOM_MASK_CENTER}" = "true" ]; then
        echo "   mask center   RANDOM per prompt, seed ${MASK_CENTER_SEED}, range +/-${MASK_CENTER_RANGE}"
        echo "                 clamped to frame: ${CLAMP_MASK_TO_FRAME} (effective range depends on radius)"
    else
        echo "   mask center   centered"
    fi
    echo "   lora          no-fov: ${NOFOV_LORA:-<none>}"
    echo "                 fov:    ${FOV_LORA:-<none>}"
    echo "   tome          merge_mlp=${MERGE_MLP}"
    echo "   ralu          ${RALU_STAGES}-stage level ${RALU_LEVEL}, compute-matched to ours@${OURS_STEPS}, z_max ${RALU_Z_MAX}"
    echo "   out root      ${OUT_ROOT}"
    local n_cfg n_arm total
    n_cfg=$(echo "${MASK_CONFIGS}" | wc -w)
    n_arm=0
    for m in ${METHODS}; do [ "$m" = "high_res" ] || n_arm=$((n_arm + 1)); done
    total=$((EXPECTED * (n_arm * n_cfg + $(echo "${METHODS}" | grep -cw high_res))))
    echo "   total images  ${total}  (${n_arm} mask-dependent arms x ${n_cfg} configs, + high_res once)"
    echo "----------------------------------------------------------------------"
    for m in ${METHODS}; do
        if [ "$m" = "high_res" ]; then
            local first; first=$(echo "${MASK_CONFIGS}" | awk '{print $1}')
            printf "   %-9s %-26s %6s pngs  %s\n" "$m" "(mask-independent)" \
                   "$(png_count "$(out_dir high_res "${first%%:*}" "${first##*:}")")" \
                   "$(out_dir high_res "${first%%:*}" "${first##*:}")"
            continue
        fi
        for cfg in ${MASK_CONFIGS}; do
            local d; d=$(out_dir "$m" "${cfg%%:*}" "${cfg##*:}")
            printf "   %-9s %-26s %6s pngs  %s\n" "$m" "$cfg" "$(png_count "$d")" "$d"
        done
    done
    echo "======================================================================"
}

verify_masks() {
    # No GPU: prove every arm foveates the identical region for the configured settings.
    # ours/naive/high_res/tome call create_foveation_mask directly; RALU works on the
    # coarse block grid and must be its exact stride-lr_factor inverse.
    $PY - <<EOF
import sys
import torch, torch.nn.functional as F
sys.path.insert(0, "${EVAL_DIR}")
sys.path.insert(0, "${EVAL_DIR}/RALU")
from inference import (create_foveation_mask, foveation_center_for_index,
                       random_mask_center_range)
from RALU_fov_inference_diffsynth import coarse_foveation_mask

H, W, LR = ${HEIGHT}, ${WIDTH}, ${LR_FACTOR}
RANDOM = "${RANDOM_MASK_CENTER}" == "true"
N = min(${EXPECTED}, 200)
print(f"{'config':18s} {'center range':>16} {'hr tokens':>10} {'m blocks':>9} {'parity':>7} {'clips':>6}")
bad = False
for cfg in "${MASK_CONFIGS}".split():
    shape, r = cfg.split(":"); r = float(r)
    rng = random_mask_center_range(${MASK_CENTER_RANGE}, r, shape, H, W, LR,
                                  clamp_to_frame=("${CLAMP_MASK_TO_FRAME}" == "true")) if RANDOM else (0.0, 0.0)
    centers = [foveation_center_for_index(i, ${MASK_CENTER_SEED}, rng) for i in range(N)] if RANDOM else [(0, 0)]
    base = int(coarse_foveation_mask(H, W, center=(0, 0), r=r, shape=shape, lr_factor=LR).sum())
    ms, ok = [], True
    for c in centers:
        tm = create_foveation_mask(H, W, center=c, r=r, shape=shape,
                                   device=torch.device("cpu"), lr_factor=LR)
        coarse = coarse_foveation_mask(H, W, center=c, r=r, shape=shape, lr_factor=LR)
        up = F.interpolate(coarse[None, None], size=tuple(tm.shape), mode="nearest")[0, 0]
        ok &= bool(torch.equal(up, tm))
        ms.append(int(coarse.sum()))
    clips = (min(ms) != base or max(ms) != base)
    bad |= (not ok) or clips
    print(f"{cfg:18s} {'(%.4f, %.4f)' % rng:>16} {int(tm.sum()):>10} "
          f"{min(ms) if min(ms)==max(ms) else '%d-%d' % (min(ms), max(ms)):>9} "
          f"{'OK' if ok else 'FAIL':>7} {'YES' if clips else 'no':>6}")
print()
print("FAILURES PRESENT" if bad else
      f"all arms share an identical mask{' at every sampled center' if RANDOM else ''}, "
      "and it never cuts the frame edge")
sys.exit(1 if bad else 0)
EOF
}

run_shard() {   # run_shard <method> <shape:radius> <shard_id> <num_shards>
    # One shard, one GPU, in the foreground: what a Slurm array element calls. The GPU comes
    # from the scheduler, and every flag comes from the same helpers the local driver uses,
    # so a cluster run and a local run are the same computation.
    local method="$1" cfg="$2" shard="$3" shards="$4"
    is_method "$method" || { echo "unknown method: $method" >&2; return 2; }
    local shape="${cfg%%:*}" radius="${cfg##*:}"
    local out; out=$(out_dir "$method" "$shape" "$radius")
    mkdir -p "$out"
    NUM_SUBSETS="$shards"
    echo "shard ${shard}/${shards}  ${method}  ${shape} r=${radius}  -> ${out}"
    "eval_${method}" "" "$shard" "$shape" "$radius" "$out"
}

case "${MODE}" in
  plan) print_plan ;;
  masks) verify_masks ;;
  shard)
    # ./eval.sh shard <method> <shape:radius> <shard_id> <num_shards>
    [ "$#" -ge 5 ] || { echo "usage: $0 shard <method> <shape:radius> <shard_id> <num_shards>" >&2; exit 1; }
    run_shard "$2" "$3" "$4" "$5"
    ;;
  check)
    for m in ${METHODS}; do
        for cfg in ${MASK_CONFIGS}; do
            d=$(out_dir "$m" "${cfg%%:*}" "${cfg##*:}")
            have=$(png_count "$d")
            meta=$(ls "$d"/metadata_*.csv 2>/dev/null | wc -l)
            status="INCOMPLETE"; [ "$have" -eq "$EXPECTED" ] && status="COMPLETE"
            printf "%-10s %-9s %-18s %6s/%-6s pngs  %s csv  %s\n" \
                   "$status" "$m" "$cfg" "$have" "$EXPECTED" "$meta" "$d"
            [ "$m" = "high_res" ] && break     # one dir, shared by all configs
        done
    done
    ;;
  all)
    print_plan
    mkdir -p "$LOG_DIR"
    for m in ${METHODS}; do run_method "$m"; done
    echo "all methods done -> ${OUT_ROOT}"
    ;;
  *)
    if is_method "${MODE}"; then
        mkdir -p "$LOG_DIR"
        run_method "${MODE}"
        echo "${MODE} done -> ${OUT_ROOT}"
    else
        echo "usage: $0 {plan|masks|check|all|high_res|naive|ours|tome|ralu}" >&2
        exit 1
    fi
    ;;
esac
