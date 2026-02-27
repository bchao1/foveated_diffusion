# Naive mixed-res inference: paths aligned with base.sh, params from eval/naive.sh
root_folder="/scratch/m000051-pm05/brianchc_2"

CUDA_VISIBLE_DEVICES=0 python /users/brianchc/foveated_diffusion/eval/inference.py \
    --experiment runtime \
    --decode_mode merge \
    --tokenAE_training false \
    --output_dir "${root_folder}/foveated_diffusion/results/runtime" \
    --seed 42 \
    --mask_shape square \
    --foveation_outline true \
    --soft_foveation_blend true