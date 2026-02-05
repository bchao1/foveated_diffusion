export DIFFSYNTH_MODEL_BASE_PATH="/miele/brian/modelscope"
cd ..
accelerate launch train.py \
  --dataset_base_path /local/brian/Aesthetic-Train-V2/ \
  --dataset_metadata_path /local/brian/Aesthetic-Train-V2/metadata.csv \
  --max_pixels 1048576 \
  --height 1024 \
  --width 1024 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image:text_encoder/*.safetensors,Tongyi-MAI/Z-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Z-Image_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --use_wandb \
  --wandb_project "z-image-training" \
  --wandb_run_name "Z-Image-lora-v3" \
  --validation_prompts \
  "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights." \
  "Golden retriever on a grassy meadow. Thick sun-kissed fur, clean and glossy, subtle lighter ruff around the neck. Alert amber eyes, black moist nose, relaxed open mouth with pink tongue. Perky ears tilted forward, sturdy athletic stance, tail held in a gentle arc. Clutches a slightly chewed tennis ball, lime green with faded white curve. Tiny firefly-like orbs drifting above the dog’s outstretched paw, soft flickering glow. Warm late-afternoon backdrop, rolling wild grass bending in the breeze, blurred tree line and distant hills under a pastel sky." \
  "Sleek black sports car on a neon city street. Polished body reflecting streaks of colored light, chrome rims gleaming. Low aerodynamic profile, tinted windows, sharp LED headlights piercing the night. Subtle blue underglow tracing the chassis, glowing beneath the doors. Holographic speed-lines shimmering above the hood, cool electric sheen. Rain-slick asphalt mirror below, tall illuminated skyscrapers rising on both sides, traffic light trails and blurred signs forming a vibrant urban haze." \
  --validation_steps 100 \
  --validation_height 1024 \
  --validation_width 1024 \
  --num_validation_images 1
cd lora
