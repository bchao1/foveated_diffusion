"""
FLUX2 VAE Encode-Decode - No DiT, no diffusion.

Loads the VAE the same way as FLUX2-foveated-inference.py and runs:
  image -> VAE encode -> latent -> VAE decode -> image

Useful for testing VAE fidelity or latent-space operations without loading the DiT.
"""

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from diffsynth.pipelines.flux2_image_foveated import Flux2FoveatedImagePipeline
from diffsynth.pipelines.flux2_image import ModelConfig


def load_vae_only(args):
    """Load pipeline with only the VAE (no DiT, no text encoder)."""
    print("Loading FLUX2 VAE (VAE only, no DiT)...")
    pipe = Flux2FoveatedImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=None,
    )
    return pipe


@torch.no_grad()
def encode_decode(pipe, image: Image.Image, upsampler):
    """Encode image to latent, then decode back to image."""
    device = pipe.device
    dtype = pipe.torch_dtype
    pipe.load_models_to_device(["vae"])
    # Preprocess: PIL -> tensor [1, C, H, W], range [-1, 1] (uses pipe.device, pipe.torch_dtype)
    image_tensor = pipe.preprocess_image(image)

    # First bring from [-1, 1] → [0, 1]
    image_tensor = (image_tensor + 1) / 2  

    # Convert mean/std to tensors and reshape for broadcasting
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)

    # Normalize per channel
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.to(device=device, dtype=dtype)
    
    # Ensure divisibility by 16 (FLUX2 latent factor)
    _, _, h, w = image_tensor.shape
    h_aligned = (h // 16) * 16
    w_aligned = (w // 16) * 16
    if h_aligned != h or w_aligned != w:
        image_tensor = torch.nn.functional.interpolate(
            image_tensor, size=(h_aligned, w_aligned), mode="bilinear", align_corners=False
        )
    # Encode (image_tensor already on device/dtype from preprocess_image)
    latents = pipe.vae.encode(image_tensor)
    print('latents.shape', latents.shape)
    exit()
    # Upsampler expects float32; move output to pipe device and dtype before decode
    latents = upsampler(image_tensor.float(), latents.float(), output_size=(latents.shape[2]*2, latents.shape[3]*2))
    latents = latents.to(device=device, dtype=dtype)
    # Decode
    decoded = pipe.vae.decode(latents)
    # Tensor -> PIL
    image_out = pipe.vae_output_to_image(decoded)
    pipe.load_models_to_device([])
    return image_out


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipe = load_vae_only(args)
    upsampler = torch.hub.load('wimmerth/anyup', 'anyup_multi_backbone', use_natten=True)
    upsampler = upsampler.to(device=pipe.device, dtype=torch.float32).eval()

    # Load input image
    if not os.path.isfile(args.input_path):
        raise FileNotFoundError(f"Input image not found: {args.input_path}")
    image = Image.open(args.input_path).convert("RGB")

    # Encode and decode
    image_out = encode_decode(pipe, image, upsampler)

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    image_out.save(args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX2 VAE encode-decode (no DiT)")
    parser.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.2-klein-base-4B",
        help="HuggingFace model ID (VAE is loaded from this repo)",
    )
    parser.add_argument("--input_path", type=str, required=True, help="Input image path")
    parser.add_argument("--output_path", type=str, default="./vae_reconstructed.png", help="Output image path")

    args = parser.parse_args()
    main(args)
