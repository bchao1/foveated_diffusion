from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch
import os

os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/miele/brian/modelscope"

pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="tokenizer/"),
)
prompt = "Documentary-style imagery: a lively little dog stands still on a lush green lawn, filling the entire frame. The dog has brownish-yellow fur, with both ears perked up, and an expression that is focused and cheerful. Sunlight evenly illuminates its body, making the fur appear soft, glossy, and highly detailed. Every part of the scene is in sharp focus, from the dog's fur to the surrounding grass and distant background elements. The background shows an open grassy field with scattered wildflowers, and faint blue sky with a few white clouds, all rendered clearly and crisply. Strong realism and clarity, emphasizing texture, color, and detail throughout the image. Full-frame composition, static pose, no depth-of-field blur, everything sharp."
image = pipe(prompt, seed=0, rand_device="cuda", num_inference_steps=50, cfg_scale=4)
image.save("image_FLUX.2-klein-base-4B.jpg")

#prompt = "change the color of the clothes to red"
#image = pipe(prompt, edit_image=[image], seed=1, rand_device="cuda", num_inference_steps=50, cfg_scale=4)
#image.save("image_edit_FLUX.2-klein-base-4B.jpg")
