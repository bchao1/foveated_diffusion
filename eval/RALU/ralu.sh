#!/bin/bash
# RALU-foveated baseline — launcher mirroring eval/ours.sh and eval/naive.sh.
#
# Both variants are SPEEDUP-MATCHED to foveated diffusion at OURS_STEPS (50): the total
# step count is solved so RALU's token-steps equal ours on the same mask, with RALU's own
# stage proportions held fixed. e and NT-DM's Z/shift are left exactly as RALU picks them.
#
#   ./ralu.sh plan                          print every derived schedule, run nothing
#   ./ralu.sh single <gpu> [stages] [r]     one prompt
#   ./ralu.sh eval   [stages] [r]           full prompt set, sharded over NUM_SUBSETS GPUs
#   ./ralu.sh sweep                         eval for every VARIANT x RADIUS, sequentially
#   ./ralu.sh sanity <gpu> [stages]         all-ones mask, validates the port
#
#   stages: 3 = their full design, uniform full-res output   (works)
#           2 = stage 3 dropped, mixed-res output like ours  (known broken, see doc)
#
# Examples
#   ./ralu.sh single 0 3 0.3
#   ./ralu.sh eval 2 0.5
#   VARIANTS="2 3" RADII="0.1 0.3 0.5" ./ralu.sh sweep
#
# See RALU_FOVEATED.md.

set -u

export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True
# imp_tome resolves `diffsynth` to a different checkout; force the private repo's copy.
export PYTHONPATH=/home/brianchc/foveated_diffusion-private/DiffSynth-Studio

PY=/home/brianchc/miniconda3/envs/imp_tome/bin/python
SCRIPT=RALU_fov_inference_diffsynth.py

# ---- configuration ---------------------------------------------------------------
# RALU is training-free, so it uses the same no-foveation LoRA as the other baselines.
step=15000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/"
LORA="${ckpt_folder}/step-${step}.safetensors"

OURS_STEPS=${OURS_STEPS:-50}        # our method's step count — the speedup match target
RALU_LEVEL=${RALU_LEVEL:-4}         # published operating point supplying stage proportions
MASK_SHAPE=${MASK_SHAPE:-circular}  # circular | square
LR_FACTOR=${LR_FACTOR:-2}           # periphery downsample factor (2 = 2x2 low-res blocks)
SEED=${SEED:-42}
CFG=${CFG:-4.0}                     # true CFG; must match eval/inference.py --guidance_scale
NUM_SUBSETS=${NUM_SUBSETS:-8}
HEIGHT=${HEIGHT:-1024}
WIDTH=${WIDTH:-1024}

# defaults for the positional args
DEF_STAGES=${DEF_STAGES:-3}
DEF_RADIUS=${DEF_RADIUS:-0.3}

# for `sweep`
VARIANTS=${VARIANTS:-"3 2"}
RADII=${RADII:-"0.1 0.3 0.5 0.7"}

PROMPTS=${PROMPTS:-/local/brian/Aesthetic-Train-V2/metadata_test.csv}
NUM_PROMPTS=${NUM_PROMPTS:-}        # empty = all
OUT_ROOT=${OUT_ROOT:-/local/brian/foveated_diffusion/results}
QUAL_OUT_ROOT=${QUAL_OUT_ROOT:-../../results/ralu_compare}

QUAL_PROMPT="A lion lounges calmly, displaying its powerful physique, with a soft focus highlighting its majestic mane and intense gaze."

# ---- helpers ---------------------------------------------------------------------
# Shared flags for a (stages, radius) configuration. Speedup matching is always on.
common_args() {
    local stages="$1" radius="$2"
    echo "--height ${HEIGHT} --width ${WIDTH} \
          --full_eval_mask ${MASK_SHAPE} --mask_radius ${radius} \
          --lr_downsample_factor ${LR_FACTOR} \
          --num_stages ${stages} --ralu_level ${RALU_LEVEL} \
          --match_compute --ours_steps ${OURS_STEPS} \
          --cfg_scale ${CFG} \
          --seed ${SEED}"
}

out_dir() {
    local stages="$1" radius="$2"
    echo "${OUT_ROOT}/ralu_${stages}stage_${MASK_SHAPE}_r${radius}_lr${LR_FACTOR}_cm${OURS_STEPS}_eval"
}

run_subset() {
    local gpu_id="$1" subset_idx="$2" stages="$3" radius="$4"
    local extra=""
    [ -n "${NUM_PROMPTS}" ] && extra="--num_prompts ${NUM_PROMPTS}"
    CUDA_VISIBLE_DEVICES=$gpu_id $PY $SCRIPT \
        --output_dir "$(out_dir "$stages" "$radius")" \
        --lora_checkpoint "${LORA}" \
        --full_eval --prompt_dataset_path "${PROMPTS}" ${extra} \
        $(common_args "$stages" "$radius") \
        --num_subsets "${NUM_SUBSETS}" --subset_idx "${subset_idx}"
}

run_eval() {
    local stages="$1" radius="$2"
    echo "=== RALU ${stages}-stage | ${MASK_SHAPE} r=${radius} lr=${LR_FACTOR} | speedup-matched to ours@${OURS_STEPS} ==="
    for i in $(seq 0 $((NUM_SUBSETS - 1))); do
        run_subset "$i" "$i" "$stages" "$radius" &
    done
    wait
    echo "done -> $(out_dir "$stages" "$radius")"
}

# ---- modes -----------------------------------------------------------------------
case "${1:-single}" in

  plan)
    $PY - <<EOF
import sys
sys.path.insert(0, ".")
from pipeline_flux2_diffsynth_RALU_fov import (
    ralu_schedule, stage_token_counts, ours_token_cost)
from RALU_fov_inference_diffsynth import coarse_foveation_mask

H, W, LR, S = ${HEIGHT}, ${WIDTH}, ${LR_FACTOR}, ${OURS_STEPS}
M = (H // 16) * (W // 16)
full_cost = S * M                     # full-res reference: S steps x M tokens
print(f"speedup-matched to ours @ {S} steps | shape=${MASK_SHAPE} lr={LR} level=${RALU_LEVEL}")
print(f"full-res reference cost = {full_cost}  ({S} steps x {M} tok)")
print()
print(f"{'r':>5} {'m':>6} {'area%':>7} | {'ours tok/st':>11} {'ours cost':>10} {'ours x':>7} "
      f"| {'st':>2} {'N':>16} {'steps':>5} {'ralu cost':>10} {'ralu x':>7} {'match':>6}")
for r in [${RADII// /, }]:
    m = int(coarse_foveation_mask(H, W, r=r, shape="${MASK_SHAPE}", lr_factor=LR).sum())
    md = M // (LR ** 2)
    oc = ours_token_cost(H, W, LR, m, S)
    for stages in [${VARIANTS// /, }]:
        N, e, info = ralu_schedule(level=${RALU_LEVEL}, num_stages=stages, match_compute=True,
                                   height=H, width=W, f0=LR, n_hr_blocks=m, ours_steps=S)
        rc = info["ralu_cost"]
        print(f"{r:5.2f} {m:6d} {100*m/md:7.2f} | {oc//S:11d} {oc:10d} {full_cost/oc:6.2f}x "
              f"| {stages:2d} {str(N):>16} {sum(N):5d} {rc:10d} {full_cost/rc:6.2f}x "
              f"{info['cost_ratio']:6.3f}")
EOF
    ;;

  single)
    gpu=${2:-0}; stages=${3:-$DEF_STAGES}; radius=${4:-$DEF_RADIUS}
    CUDA_VISIBLE_DEVICES=$gpu $PY $SCRIPT \
        --output_dir "${QUAL_OUT_ROOT}/ralu_${stages}stage_r${radius}_cm${OURS_STEPS}" \
        --lora_checkpoint "${LORA}" \
        --prompt "${QUAL_PROMPT}" \
        $(common_args "$stages" "$radius")
    ;;

  eval)
    run_eval "${2:-$DEF_STAGES}" "${3:-$DEF_RADIUS}"
    ;;

  sweep)
    for stages in ${VARIANTS}; do
        for radius in ${RADII}; do
            run_eval "$stages" "$radius"
        done
    done
    ;;

  sanity)
    gpu=${2:-0}; stages=${3:-$DEF_STAGES}
    CUDA_VISIBLE_DEVICES=$gpu $PY $SCRIPT \
        --output_dir "${QUAL_OUT_ROOT}/ralu_sanity_${stages}stage" \
        --lora_checkpoint "${LORA}" \
        --prompt "${QUAL_PROMPT}" \
        --num_inference_steps "${OURS_STEPS}" \
        --num_stages "${stages}" --ralu_level "${RALU_LEVEL}" \
        --cfg_scale "${CFG}" \
        --seed "${SEED}" --full_res_check
    ;;

  *)
    echo "usage: $0 {plan|single|eval|sweep|sanity} [args]" >&2
    echo "  single <gpu> [stages] [radius]" >&2
    echo "  eval   [stages] [radius]" >&2
    exit 1
    ;;
esac
