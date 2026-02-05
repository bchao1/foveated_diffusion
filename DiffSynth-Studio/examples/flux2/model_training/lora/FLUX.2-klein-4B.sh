export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
cd ..
accelerate launch --gpu_ids $1 train.py \
  --dataset_base_path /local/brian/Aesthetic-Train-V2/ \
  --dataset_metadata_path /local/brian/Aesthetic-Train-V2/metadata.csv \
  --max_pixels 1048576 \
  --height 1024 \
  --width 1024 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-4B:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "black-forest-labs/FLUX.2-klein-4B:tokenizer/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/FLUX.2-klein-4B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --use_wandb \
  --wandb_project "flux2-training" \
  --wandb_run_name "FLUX.2-klein-4B-lora-v1" \
  --validation_prompts \
  "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp, bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda, blurred colorful distant lights." \
  "Golden retriever on a grassy meadow. Thick sun-kissed fur, clean and glossy, subtle lighter ruff around the neck. Alert amber eyes, black moist nose, relaxed open mouth with pink tongue. Perky ears tilted forward, sturdy athletic stance, tail held in a gentle arc. Clutches a slightly chewed tennis ball, lime green with faded white curve. Tiny firefly-like orbs drifting above the dog's outstretched paw, soft flickering glow. Warm late-afternoon backdrop, rolling wild grass bending in the breeze, blurred tree line and distant hills under a pastel sky." \
  "Sleek black sports car on a neon city street. Polished body reflecting streaks of colored light, chrome rims gleaming. Low aerodynamic profile, tinted windows, sharp LED headlights piercing the night. Subtle blue underglow tracing the chassis, glowing beneath the doors. Holographic speed-lines shimmering above the hood, cool electric sheen. Rain-slick asphalt mirror below, tall illuminated skyscrapers rising on both sides, traffic light trails and blurred signs forming a vibrant urban haze." \
  --validation_steps 100 \
  --validation_height 1024 \
  --validation_width 1024 \
  --num_validation_images 1
cd lora
