exp_name="no_tokenAE_fix_mask"
root_folder="/scratch/m000051-pm05/brianchc_2"

accelerate launch train_foveated.py \
  --config_file /users/brianchc/accelerate_config.yaml \
  /users/brianchc/foveated_diffusion/DiffSynth-Studio/examples/flux2/model_training/train_foveated.py \
  --dataset_base_path "${root_folder}/Aesthetic-Train-V2" \
  --dataset_metadata_path "${root_folder}/Aesthetic-Train-V2/metadata_train.csv" \
  --max_pixels 1048576 \
  --height 1024 \
  --width 1024 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-base-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "black-forest-labs/FLUX.2-klein-base-4B:tokenizer/" \
  --learning_rate 1e-4 \
  --tokenAE_learning_rate 1e-4 \
  --tokenAE_train_steps 0 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${root_folder}/foveated_diffusion/experiments/${exp_name}" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --use_wandb \
  --wandb_project "flux2-training" \
  --wandb_run_name $exp_name \
  --validation_prompts \
  "Golden retriever on a grassy meadow. Thick sun-kissed fur, clean and glossy, subtle lighter ruff around the neck. Alert amber eyes, black moist nose, relaxed open mouth with pink tongue. Perky ears tilted forward, sturdy athletic stance, tail held in a gentle arc. Warm late-afternoon backdrop, rolling wild grass bending in the breeze, blurred tree line and distant hills under a pastel sky." \
  --validation_steps 100 \
  --validation_height 1024 \
  --validation_width 1024 \
  --num_validation_images 1 \
  --save_steps 1000 \
  --task sft_no_tokenAE \
  --find_unused_parameters \
  --decode_mode "merge" \
  --foveated_training_mode fixed
cd lora