#!/bin/bash
# Qualitative ours-vs-RALU grid across mask radii, at cfg_scale 4.0 (see rerun_cfg4.sh).
# One prompt; ours (mixed-res, 50 steps) vs RALU 3-stage / 2-stage, compute-matched at
# each r, with NT-DM's Z both free and clamped to their re-noising strength.
set -u

export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTHONPATH=/home/brianchc/foveated_diffusion-private/DiffSynth-Studio
PY=/home/brianchc/miniconda3/envs/imp_tome/bin/python

PROMPT="A lion lounges calmly, displaying its powerful physique, with a soft focus highlighting its majestic mane and intense gaze."
RADII=${RADII:-"0.1 0.3 0.5 0.7"}
OUT=/home/brianchc/foveated_diffusion-private/results/ralu_compare/cfg4/grid
SEED=42
LR=2
CFG=4.0
GPUS=(0 6 7 8 9)

FOV_LORA=/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-5000.safetensors
NOFOV_LORA=/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/step-15000.safetensors

mkdir -p "$OUT"

for r in $RADII; do
    g=0
    CUDA_VISIBLE_DEVICES=${GPUS[$g]} $PY ../inference.py \
        --experiment ours --decode_mode merge --tokenAE_training false \
        --lora_checkpoint "$FOV_LORA" \
        --full_eval_mask circular --mask_radius "$r" --lr_downsample_factor $LR \
        --soft_foveation_blend true --foveation_outline false \
        --num_inference_steps 50 --guidance_scale $CFG \
        --prompt "$PROMPT" --seed $SEED \
        --output_dir "$OUT/ours_r${r}" > "$OUT/ours_r${r}.log" 2>&1 &

    for st in 3 2; do
        for zm in 100 6.5; do
            g=$((g+1))
            tag="ralu${st}_z${zm}_r${r}"
            CUDA_VISIBLE_DEVICES=${GPUS[$((g % 5))]} $PY RALU_fov_inference_diffsynth.py \
                --prompt "$PROMPT" --output_dir "$OUT/$tag" \
                --lora_checkpoint "$NOFOV_LORA" \
                --full_eval_mask circular --mask_radius "$r" --lr_downsample_factor $LR \
                --num_stages $st --ralu_level 4 --match_compute --ours_steps 50 \
                --z_max $zm --cfg_scale $CFG \
                --seed $SEED > "$OUT/$tag.log" 2>&1 &
        done
    done
    wait
    echo "r=$r done"
done

# Radius-independent references on the same prompt: full-res, and RALU's published config.
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY ../inference.py \
    --experiment high_res --decode_mode direct --tokenAE_training false \
    --lora_checkpoint "$NOFOV_LORA" \
    --num_inference_steps 50 --guidance_scale $CFG \
    --prompt "$PROMPT" --seed $SEED \
    --output_dir "$OUT/plain" > "$OUT/plain.log" 2>&1 &

CUDA_VISIBLE_DEVICES=${GPUS[1]} $PY RALU_fov_inference_diffsynth.py \
    --prompt "$PROMPT" --output_dir "$OUT/ralu3_pub_r0.3" \
    --lora_checkpoint "$NOFOV_LORA" \
    --full_eval_mask circular --mask_radius 0.3 --lr_downsample_factor $LR \
    --num_stages 3 --ralu_level 4 --use_published \
    --cfg_scale $CFG --seed $SEED > "$OUT/ralu3_pub_r0.3.log" 2>&1 &
wait
echo "grid done -> $OUT"
