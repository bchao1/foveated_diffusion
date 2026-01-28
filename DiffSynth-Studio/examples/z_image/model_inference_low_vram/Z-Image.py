import os
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
import torch

model_folder = "/miele/brian/modelscope/Z-Image"
model_folder = "/scratch/m000051-pm04/brianchc/Z-Image"

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=[
            os.path.join(model_folder, "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"),
            os.path.join(model_folder, "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"),
        ], **vram_config),
        ModelConfig(path=[
            os.path.join(model_folder, "text_encoder/model-00001-of-00003.safetensors"),
            os.path.join(model_folder, "text_encoder/model-00002-of-00003.safetensors"),
            os.path.join(model_folder, "text_encoder/model-00003-of-00003.safetensors"),
        ], **vram_config),
        ModelConfig(path=os.path.join(model_folder, "vae/diffusion_pytorch_model.safetensors"), **vram_config),
    ],
    tokenizer_config=ModelConfig(path=os.path.join(model_folder, "tokenizer/"), **vram_config),
)
prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
image = pipe(prompt=prompt, seed=42, rand_device="cuda", num_inference_steps=50, cfg_scale=4)
image.save("image_Z-Image.jpg")
