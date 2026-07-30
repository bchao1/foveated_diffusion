#!/bin/bash
# Generate the ours-vs-RALU summary grid on one prompt across mask radii.
# ours (mixed-res, 50 steps) vs RALU 3-stage and 2-stage, both speedup-matched at each r.
set -u

export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTHONPATH=/home/brianchc/foveated_diffusion-private/DiffSynth-Studio
PY=/home/brianchc/miniconda3/envs/imp_tome/bin/python

PROMPT="A lion lounges calmly, displaying its powerful physique, with a soft focus highlighting its majestic mane and intense gaze."
RADII="0.1 0.3 0.5 0.7"
OUT=/home/brianchc/foveated_diffusion-private/results/ralu_compare/grid
SEED=42
LR=2

FOV_LORA=/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-5000.safetensors
NOFOV_LORA=/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/step-15000.safetensors

mkdir -p "$OUT"
# NOTE: assign in the parent shell -- $(fn) runs in a subshell and would not persist g.
g=0

for r in $RADII; do
    # ours
    gpu=$((g % 10)); g=$((g+1))
    CUDA_VISIBLE_DEVICES=$gpu $PY ../inference.py \
        --experiment ours --decode_mode merge --tokenAE_training false \
        --lora_checkpoint "$FOV_LORA" \
        --full_eval_mask circular --mask_radius "$r" --lr_downsample_factor $LR \
        --soft_foveation_blend true --foveation_outline false \
        --prompt "$PROMPT" --seed $SEED \
        --output_dir "$OUT/ours_r${r}" > "$OUT/ours_r${r}.log" 2>&1 &

    # RALU, both variants, speedup-matched to ours@50 at this radius
    for st in 3 2; do
        gpu=$((g % 10)); g=$((g+1))
        CUDA_VISIBLE_DEVICES=$gpu $PY RALU_fov_inference_diffsynth.py \
            --prompt "$PROMPT" --output_dir "$OUT/ralu${st}_r${r}" \
            --lora_checkpoint "$NOFOV_LORA" \
            --full_eval_mask circular --mask_radius "$r" --lr_downsample_factor $LR \
            --num_stages $st --ralu_level 4 --match_compute --ours_steps 50 \
            --seed $SEED > "$OUT/ralu${st}_r${r}.log" 2>&1 &
    done
done
wait
echo "all runs done -> $OUT"
