#!/bin/bash
# Token Autoencoder Training Script
# Trains TokenDownsampler and TokenUpsampler to compress and reconstruct VAE latents

accelerate launch --main_process_port 29503 --num_processes 8 train_tokenAE.py \
  --pretrained_model_name_or_path="/scratch/m000051-pm04/brianchc/FLUX.1-dev" \
  --instance_data_dir="/scratch/m000051-pm04/brianchc/Aesthetic-Train-V2" \
  --output_dir="/scratch/m000051-pm04/brianchc/flux_foveation/model_checkpoints/tokenAE" \
  --mixed_precision="bf16" \
  --train_batch_size=4 \
  --resolution 1024 \
  --dataloader_num_workers=4 \
  --gradient_accumulation_steps=2 \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --adam_weight_decay=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=1000 \
  --max_train_steps=100000 \
  --seed="0" \
  --tokenAE_feature_channels 64 \
  --downsampler_num_res_blocks 2 \
  --upsampler_num_res_blocks 10 \
  --downsampler_learn_residual \
  --upsampler_learn_residual \
  --naive_upsample_mode "nearest" \
  --naive_downsample_mode "nearest" \
  --log_image_steps 100
