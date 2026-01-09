import os
import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import (
    CLIPTokenizer, 
    T5TokenizerFast, 
    CLIPTextModel, 
    T5EncoderModel
)

# Use the exact class from your local file
from transformer_flux_crpa import FluxPhaseAlignedTransformer2DModel 
from pipeline_flux_foveation_tokenAE import FluxPipeline_Foveation
from utils import create_foveation_mask
from adapter_foveation import TokenDownsampler, TokenUpsampler

## 1. Setup Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.bfloat16
model_id = "black-forest-labs/FLUX.1-dev"
resolution = 1024
batch_size = 1
max_sequence_length = 256

print(f"Testing on device: {device} with dtype: {weight_dtype}")

## 2. Load Models (Matching your Pipeline __init__)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
tokenizer_2 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2")

# Note: Standard CLIPTextModel used in your pipeline script
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device, dtype=weight_dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2").to(device, dtype=weight_dtype)

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device, dtype=weight_dtype)
# Replace with FluxPhaseAlignedTransformer2DModel if testing your custom class
transformer = FluxPhaseAlignedTransformer2DModel.from_pretrained(model_id, subfolder="transformer").to(device, dtype=weight_dtype)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
#pipe = FluxPipeline_Foveation.from_pretrained(model_id, transformer=transformer, torch_dtype=weight_dtype).to(device, dtype=weight_dtype)

# foveation mask logic. dummy foveation mask
foveation_mask = create_foveation_mask(resolution // 16, resolution // 16, center=(0, 0), r = 0.1) # low-res foveation mask then upsample
print(f"foveation_mask shape: {foveation_mask.shape}")

#foveation_mask_low_res = foveation_mask.clone()
#foveation_mask = foveation_mask.unsqueeze(0).unsqueeze(0).float() # [1, 1, H, W]
#foveation_mask_high_res = F.interpolate(foveation_mask, size=(resolution // 16, resolution // 16), mode='nearest')
#foveation_mask_high_res = foveation_mask_high_res.squeeze()
#foveation_mask = foveation_mask_high_res.to(device, dtype=weight_dtype) # 64, 64

# set token_upsampler and token_downsampler attributes in pipe
token_upsampler = TokenUpsampler(in_channels=64, out_channels=64, num_res_blocks=10).to(device).bfloat16()
token_downsampler = TokenDownsampler(in_channels=64, out_channels=64, num_res_blocks=10).to(device).bfloat16()



def get_clip_prompt_embeds(prompt, batch_size):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77, # tokenizer_max_length
        truncation=True,
        return_tensors="pt",
    ).to(device)
    
    outputs = text_encoder(text_inputs.input_ids, output_hidden_states=False)
    
    # FIX: If CLIPTextModel doesn't have pooler_output, use the hidden state of the EOS token
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        prompt_embeds = outputs.pooler_output
    else:
        # Standard fallback for CLIPTextModel (non-projection)
        prompt_embeds = outputs.last_hidden_state[:, text_inputs.input_ids.argmax(dim=-1)]
    
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    return prompt_embeds.view(batch_size, -1)

def get_t5_prompt_embeds(prompt, batch_size):
    text_inputs = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    
    prompt_embeds = text_encoder_2(text_inputs.input_ids, output_hidden_states=False)[0]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder_2.dtype, device=device)
    return prompt_embeds.view(batch_size, max_sequence_length, -1)

def encode_prompt(prompt):
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    pooled_prompt_embeds = get_clip_prompt_embeds(prompt, batch_size)
    prompt_embeds = get_t5_prompt_embeds(prompt, batch_size)
    
    # Matching your pipeline's text_ids creation
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=weight_dtype)
    return prompt_embeds, pooled_prompt_embeds, text_ids

## 4. Testing Forward Pass
print("\nRunning Forward Pass...")
prompt = "a professional photograph of a futuristic city"
joint_attention_kwargs = {}

with torch.no_grad():
    # Encode text prompts
    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(prompt)

    # VAE encoding
    pixel_values = torch.randn(batch_size, 3, 1024, 1024).to(device, dtype=weight_dtype)
    model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
    model_input = model_input.to(dtype=weight_dtype) # Line 922 in inference, shape (1, 16, 128, 128), latents

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) # Line 211 in inference

    # downsample latents for mixed-resolution tokens
    bsz, latent_height, latent_width = model_input.shape[0], model_input.shape[2] // 2, model_input.shape[3] // 2 # (64, 64)
    latent_image_ids, resolution_mask, resolution_mask_top_left = FluxPipeline_Foveation._prepare_latent_image_ids(
        bsz, latent_height, latent_width, device, weight_dtype, foveation_mask
    )
    joint_attention_kwargs.update({"resolution_mask": resolution_mask, "resolution_mask_top_left": resolution_mask_top_left})

    # pack model input and add noise (no noise here)
    # add noise here and pack
    noise = torch.randn_like(model_input)
    noisy_model_input = model_input + noise # fix noise here
    # pack and downsample latents
    packed_noisy_model_input = FluxPipeline_Foveation._pack_latents(noisy_model_input, model_input.shape[0], model_input.shape[1], model_input.shape[2], model_input.shape[3]) # Line 924. (1, 4096, 64)
    packed_noisy_model_input = FluxPipeline_Foveation._downsample_latents(packed_noisy_model_input, latent_height, latent_width, foveation_mask=foveation_mask, token_downsampler=token_downsampler) # Line 1265, (1, 1087, 64)

    guidance = torch.tensor([3.5]).to(device, dtype=weight_dtype).expand(bsz) # Line 1213
    timesteps = torch.tensor([1.0]).to(device, dtype=weight_dtype).expand(bsz) # Line 1261

    model_pred = transformer(
        hidden_states=packed_noisy_model_input,
        # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
        timestep=timesteps / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        joint_attention_kwargs=joint_attention_kwargs,
        return_dict=False,
    )[0] # (1, 1087, 64) # Line 1272
    model_pred = FluxPipeline_Foveation._upsample_noise_pred(model_pred, latent_height, latent_width, foveation_mask, token_upsampler=token_upsampler) # Line 1283, (1, 4096, 64)
    model_pred = FluxPipeline_Foveation._unpack_latents(
        model_pred,
        height=model_input.shape[2] * vae_scale_factor,
        width=model_input.shape[3] * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    ) # (1, 16, 128, 128)

    assert model_pred.shape == noisy_model_input.shape
    
    


    print(f"SUCCESS! Output noise_pred shape: {model_pred.shape}")