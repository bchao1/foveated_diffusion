import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from pipe_pixart import PixArtAlphaPipeline 
from pathlib import Path

device = torch.device("cuda:0")
Path("results").mkdir(parents=True, exist_ok=True)

model_id = "PixArt-alpha/PixArt-XL-2-1024-MS"
pipe = PixArtAlphaPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

ratio, seed, prompt = 0.5, 50, "A cute cat"
# ratio, seed, prompt = 0.5, 42, "A real beautiful face."
# ratio, seed, prompt = 0.5, 42, "a small cactus with a happy face in the Sahara desert" 

num_inference_steps = 20
prune_from_i, merge_from_i = 0, 4

image = pipe(
    prompt=prompt, generator=torch.manual_seed(seed), 
    num_inference_steps=num_inference_steps).images[0]
image.save('results/base.png')

image = pipe(
    prompt=prompt, generator=torch.manual_seed(seed), 
    num_inference_steps=num_inference_steps, prune_from_i=prune_from_i, merge_from_i=merge_from_i, compress_ratio=ratio, free_err_mask=True).images[0]
image.save('results/ours.png')

image = pipe(
    prompt=prompt, generator=torch.manual_seed(seed), 
    num_inference_steps=num_inference_steps, prune_from_i=prune_from_i, merge_from_i=merge_from_i, compress_ratio=ratio).images[0]
image.save('results/tome.png')