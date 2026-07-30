#!/bin/bash
# Re-run of the compute-matched RALU-vs-ours comparison with the guidance bug fixed.
#
# Everything here now runs at cfg_scale 4.0, matching eval/inference.py's --guidance_scale
# default. The previous numbers in RALU_FOVEATED.md had RALU at cfg_scale 1.0 against
# guided baselines. See the Guidance section of that doc.
#
# Configs (all: circular r=0.3, lr_downsample_factor 2, seed 42, 16 prompts):
#   plain       full-res FLUX.2, 50 steps                      -- reference
#   ours        foveated diffusion, 50 steps                   -- the method
#   ralu3_free  3-stage, compute-matched@50, NT-DM free solve  -- doc default (Z~28)
#   ralu3_z6.5  3-stage, compute-matched@50, Z clamped to 6.5  -- their re-noising strength
#   ralu2_free  2-stage, compute-matched@50, free solve
#   ralu2_z6.5  2-stage, compute-matched@50, Z clamped to 6.5
#   ralu3_pub   their published level-4 config verbatim        -- not compute-matched
set -u

export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTHONPATH=/home/brianchc/foveated_diffusion-private/DiffSynth-Studio
PY=/home/brianchc/miniconda3/envs/imp_tome/bin/python

HERE=/home/brianchc/foveated_diffusion-private/eval/RALU
OUT=/home/brianchc/foveated_diffusion-private/results/ralu_compare/cfg4
PROMPTS=/home/brianchc/foveated_diffusion-private/eval/prompts/claude_prompts.csv
NP=16
R=0.3
LR=2
SEED=42
CFG=4.0
STEPS=50

FOV_LORA=/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-5000.safetensors
NOFOV_LORA=/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/step-15000.safetensors

mkdir -p "$OUT"

ralu() {  # ralu <name> <gpu> <extra args...>
    local name="$1" gpu="$2"; shift 2
    CUDA_VISIBLE_DEVICES=$gpu $PY "$HERE/RALU_fov_inference_diffsynth.py" \
        --output_dir "$OUT/$name" --lora_checkpoint "$NOFOV_LORA" \
        --full_eval --prompt_dataset_path "$PROMPTS" --num_prompts $NP \
        --full_eval_mask circular --mask_radius $R --lr_downsample_factor $LR \
        --ralu_level 4 --cfg_scale $CFG --seed $SEED \
        "$@" > "$OUT/$name.log" 2>&1
}

ours_run() {  # ours_run <name> <gpu> <experiment> <lora> <decode>
    local name="$1" gpu="$2" exp="$3" lora="$4" dec="$5"
    CUDA_VISIBLE_DEVICES=$gpu $PY ../inference.py \
        --experiment "$exp" --decode_mode "$dec" --tokenAE_training false \
        --output_dir "$OUT/$name" --lora_checkpoint "$lora" \
        --full_eval --prompt_dataset_path "$PROMPTS" --num_prompts $NP \
        --full_eval_mask circular --mask_radius $R --lr_downsample_factor $LR \
        --soft_foveation_blend true --foveation_outline false \
        --num_inference_steps $STEPS --guidance_scale $CFG --seed $SEED \
        > "$OUT/$name.log" 2>&1
}

cd "$HERE"

# ---- wave 1 -----------------------------------------------------------------------
ours_run plain      0 high_res "$NOFOV_LORA" direct &
ours_run ours       6 ours     "$FOV_LORA"   merge  &
ralu ralu3_free     7 --num_stages 3 --match_compute --ours_steps $STEPS &
ralu ralu3_z6.5     8 --num_stages 3 --match_compute --ours_steps $STEPS --z_max 6.5 &
ralu ralu2_free     9 --num_stages 2 --match_compute --ours_steps $STEPS &
wait
echo "wave 1 done"

# ---- wave 2 -----------------------------------------------------------------------
ralu ralu2_z6.5     0 --num_stages 2 --match_compute --ours_steps $STEPS --z_max 6.5 &
ralu ralu3_pub      6 --num_stages 3 --use_published &
wait
echo "wave 2 done -> $OUT"
