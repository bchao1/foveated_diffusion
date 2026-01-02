accelerate launch --main_process_port 29502 --num_processes 8 train_flux_foveation.py \
  --pretrained_model_name_or_path="/scratch/m000051-pm04/brianchc/FLUX.1-dev" \
  --instance_data_dir="/scratch/m000051-pm04/brianchc/Aesthetic-Train-V2" \
  --output_dir="/scratch/m000051-pm04/brianchc/flux_foveation/model_checkpoints/0102_test_8gpu" \
  --logging_dir="/scratch/m000051-pm04/brianchc/flux_foveation/logs" \
  --mixed_precision="bf16" \
  --train_batch_size=2 \
  --resolution 1024 \
  --dataloader_num_workers=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --adam_weight_decay=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=1000 \
  --max_train_steps=100000 \
  --validation_prompt "A highly detailed photograph of a racecar parked on a road, cinematic composition" \
  --seed="0"

# check diffusers version (validation image, offload, number of images with no prompts.... check FLUX finetuning scripts)
# request more RAM