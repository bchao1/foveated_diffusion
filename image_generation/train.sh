accelerate launch --main_process_port 29501 --num_processes 2 train_flux_foveation.py \
  --pretrained_model_name_or_path="/scratch/m000051-pm04/brianchc/FLUX.1-dev" \
  --dataset_name="/scratch/m000051-pm04/brianchc/Aesthetic-Train-V2" \
  --output_dir="../results/train_0102" \
  --mixed_precision="bf16" \
  --train_batch_size=2 \
  --dataloader_num_workers=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-5 \
  --adam_weight_decay=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=1000 \
  --checkpointing_steps=1000 \
  --max_train_steps=10000 \
  --validation_prompt "A highly detailed photograph of a racecar parked on a road, cinematic composition" \
  --seed="0" \
  --cache_dir "/scratch/m000051-pm04/brianchc/hf_datasets_decompressed"
#  --push_to_hub

# check diffusers version (validation image, offload, number of images with no prompts.... check FLUX finetuning scripts)
# request more RAM