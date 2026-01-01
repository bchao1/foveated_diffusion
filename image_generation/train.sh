accelerate launch --main_process_port 29503 --num_processes 4 train_flux_foveation.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --dataset_name="zhang0jhon/Aesthetic-Train-V2" \
  --output_dir="results/test_1228" \
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
  --instance_prompt "A highly detailed photograph of a racecar parked on a road, cinematic composition" \
  --cache_dir "/miele/brian/data/hf_datasets/Aesthetic-Train-V2"
#  --push_to_hub

# check diffusers version (validation image, offload, number of images with no prompts.... check FLUX finetuning scripts)