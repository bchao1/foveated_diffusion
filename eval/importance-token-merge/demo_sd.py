import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from pathlib import Path
from diffusers import DDIMScheduler
from pipe_sd import StableDiffusionPipeline

device = torch.device("cuda:0")
Path("results").mkdir(parents=True, exist_ok=True)

model_id = "stabilityai/stable-diffusion-2"
ddim = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, scheduler=ddim)
pipe.to(device)

# ratio, seed, prompt = 0.7, 0, "Young musician playing guitar on stage"
ratio, seed, prompt = 0.7, 1, "Elegant teacup with a delicate floral pattern"
# ratio, seed, prompt = 0.7, 2, "Royal castle with golden towers at sunrise"

prune_from_i, merge_from_i = 0, 2

result = pipe(prompt=prompt, generator=torch.manual_seed(seed))
result.images[0].save('results/base.png')

result = pipe(prompt=prompt, generator=torch.manual_seed(seed), prune_from_i=prune_from_i, merge_from_i=merge_from_i, compress_ratio=ratio, free_err_mask=True)
result.images[0].save('results/ours.png')

result = pipe(prompt=prompt, generator=torch.manual_seed(seed), prune_from_i=prune_from_i, merge_from_i=merge_from_i, compress_ratio=ratio)
result.images[0].save('results/tome.png')