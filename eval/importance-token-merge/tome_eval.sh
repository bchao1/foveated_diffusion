#!/bin/bash
# Distributed full-prompt eval for the ToMe-2x2 baseline, sharded 8 ways like eval/ours.sh.
# Writes img_%010d.png + metadata_%05d.csv, so eval/compute_clip_score.py,
# eval/compute_hpsv2_score.py and eval/extract_inception_features.py consume it unchanged.
#
# Usage: ./tome_eval.sh [merge_mlp]     # merge_mlp = true|false, default false
export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTHONPATH=/home/brianchc/foveated_diffusion-private/DiffSynth-Studio

MERGE_MLP=${1:-false}
OUT=../../results/tome_eval_mlp_${MERGE_MLP}
PY=/home/brianchc/miniconda3/envs/imp_tome/bin/python

nofov_lora="/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/step-15000.safetensors"
PROMPTS=../prompts/claude_prompts.csv
NUM_PROMPTS=1000

for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i $PY tome_fov_inference.py \
        --full_eval \
        --prompt_dataset_path ${PROMPTS} \
        --num_prompts ${NUM_PROMPTS} \
        --num_subsets 8 \
        --subset_idx $i \
        --output_dir ${OUT} \
        --seed 42 \
        --lora_checkpoint ${nofov_lora} \
        --full_eval_mask circular \
        --mask_radius 0.3 \
        --lr_downsample_factor 2 \
        --merge_mlp ${MERGE_MLP} &
done
wait

echo "Done. Results in ${OUT}"
