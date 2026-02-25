export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD=True

step=15000
ckpt_folder="/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/" # new, no tokenAE
lora_checkpoint="${ckpt_folder}/step-${step}.safetensors"


CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment high_res \
    --decode_mode direct \
    --output_dir ../results/eccv2026_final_v2/teaser \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --prompt "A male lion stands facing the camera in the golden light of the African savanna, his full mane cascading in thick waves of dark brown and amber from the crown of his head to his broad chest. Every strand of the mane is distinct, wind-lifted on one side, matted and heavy on the other. His wide-set amber eyes are calm and direct, nostrils slightly flared, the bridge of his nose scarred and sun-weathered. His jaw is massive and squared, the mouth closed, and the symmetry of his face is framed perfectly by the voluminous mane. The background is a soft wash of dry golden grass and a single acacia tree, and the low sun backlights the outer edges of the mane in a fiery halo of individual backlit hairs."
exit

CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --experiment high_res \
    --decode_mode direct \
    --output_dir ../results/eccv2026_final_v2/baseline_qualitative \
    --seed 42 \
    --tokenAE_training false \
    --lora_checkpoint ${lora_checkpoint} \
    --full_eval \
    --prompt_dataset_path /home/brianchc/foveated_diffusion/eval/prompts/claude_prompts.csv \
    --num_prompts 100