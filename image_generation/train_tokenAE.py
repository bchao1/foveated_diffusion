"""
Token Autoencoder Training Script

Trains a TokenDownsampler and TokenUpsampler to compress and reconstruct VAE latents.
Uses L2 loss computed directly on the latent space.
"""

import os
import sys
sys.path.append("..")

import argparse
import copy
import gc
import itertools
import logging

import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image, UnidentifiedImageError, PngImagePlugin, ImageFile

from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)

from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

# Token upsampler and downsampler
from adapter_foveation_time_cond import TokenDownsampler, TokenUpsampler

# Handle image loading
PngImagePlugin.MAX_TEXT_CHUNK = 1000 * (1024**2)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed.
check_min_version("0.36.0")

logger = get_logger(__name__)

if is_torch_npu_available():
    import torch_npu

    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)


class TokenAEWrapper(nn.Module):
    """Wrapper to hold the token upsampler and downsampler for accelerate."""
    def __init__(self, token_upsampler, token_downsampler):
        super().__init__()
        self.token_upsampler = token_upsampler
        self.token_downsampler = token_downsampler
    
    def forward(self, x, temb):
        """Forward pass: encode then decode."""
        encoded = self.token_downsampler(x, temb)
        decoded = self.token_upsampler(encoded, temb)
        return decoded, encoded


def load_text_encoders(args, class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def tensor_to_pil_latent(latent_tensor):
    """
    Converts a latent tensor [B, C, H, W] to a list of PIL Images.
    Visualizes the first 3 channels mapped to RGB.
    """
    viz = latent_tensor[:, :3, :, :].detach().float().cpu()

    images = []
    for i in range(viz.shape[0]):
        max_val = torch.max(torch.abs(viz[i]))
        if max_val > 1e-6:
            img_tensor = (viz[i] / (2 * max_val)) + 0.5
        else:
            img_tensor = torch.full_like(viz[i], 0.5)

        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        images.append(Image.fromarray(img_np))
        
    return images


def tensor_to_pil_image(image_tensor):
    """
    Expects [B, C, H, W] in range [-1, 1]
    """
    viz = image_tensor.detach().float().cpu()
    viz = (viz / 2) + 0.5
    viz = torch.clamp(viz, 0, 1)
    
    images = []
    for i in range(viz.shape[0]):
        img_np = (viz[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(img_np))
        
    return images

    
def log_images(accelerator, images_dict, is_final_validation=False):
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"

        log_payload = {}

        for image_name, image_list in images_dict.items():
            key = f"{phase_name}/{image_name}"
            log_payload[key] = [
                wandb.Image(image, caption=f"{image_name} {i}") 
                for i, image in enumerate(image_list)
            ]
        
        if tracker.name == "wandb":
            tracker.log(log_payload)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Token Autoencoder training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with the T5 text encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="token-ae",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for time embedding (used in Flux)",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--debug_mode", default=None, type=str)

    # Token AE architecture params
    parser.add_argument("--tokenAE_feature_channels", type=int, default=64, help="Feature channels in token AE")
    parser.add_argument("--downsampler_num_res_blocks", type=int, default=10, help="Number of residual blocks in token downsampler")
    parser.add_argument("--upsampler_num_res_blocks", type=int, default=10, help="Number of residual blocks in token upsampler")
    parser.add_argument("--downsampler_learn_residual", action="store_true", help="Learn residual in downsampler")
    parser.add_argument("--upsampler_learn_residual", action="store_true", help="Learn residual in upsampler")
    parser.add_argument("--use_ada_norm", action="store_true", help="Use adaptive normalization conditioned on time embedding")
    parser.add_argument("--cnn_upsample_mode", type=str, default="bilinear", choices=["bilinear", "conv"], help="Upsample mode in CNN")
    parser.add_argument("--naive_upsample_mode", type=str, default="nearest", choices=["bilinear", "nearest"], help="Naive upsample mode")
    parser.add_argument("--naive_downsample_mode", type=str, default="nearest", choices=["bilinear", "nearest"], help="Naive downsample mode")
    
    # Validation
    parser.add_argument("--validation_steps", type=int, default=100, help="Run validation every X steps")
    parser.add_argument("--log_image_steps", type=int, default=100, help="Log images every X steps")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.instance_data_dir is None:
        raise ValueError("Specify `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class CustomDataset(Dataset):
    """
    A dataset to prepare the instance images with the prompts for training.
    """

    def __init__(
        self,
        data_root,
        size=1024,
        center_crop=False,
        debug_mode=None
    ):
        self.data_root = data_root
        self.size = size
        self.center_crop = center_crop

        self.filenames = []
        self.prompts = []

        if debug_mode == "single":
            self.train_transforms = transforms.Compose(
                [
                    transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        else:
            self.train_transforms = transforms.Compose(
                [
                    transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(self.size) if self.center_crop else transforms.RandomCrop(self.size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        
        import json
        with open(os.path.join(self.data_root, "metadata.jsonl"), "r") as json_file:
            json_list = list(json_file)

            json_list.sort()
            for json_str in json_list:
                example = json.loads(json_str)
                self.filenames.append(example["file_name"])
                self.prompts.append(example["text"])

        if debug_mode == "single":
            self.filenames = [self.filenames[0]] * len(json_list)
            self.prompts = [self.prompts[0]] * len(json_list)
            print(f"Debug mode: using single image with prompt: {self.prompts[0]}")

        self._length = len(self.filenames)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        for _ in range(5):
            try:
                file_name = self.filenames[index]
                image = Image.open(os.path.join(self.data_root, file_name))
                image = exif_transpose(image)
                image = image.convert("RGB")
                pixel_values = self.train_transforms(image)
    
                return {
                    "pixel_values": pixel_values,
                    "prompt": self.prompts[index],
                }
    
            except (UnidentifiedImageError, OSError, ValueError) as e:
                index = random.randint(0, self._length - 1)


def collate_fn(examples):
    pixel_values = [example["pixel_values"] for example in examples]
    prompts = [example["prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    return batch


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    print("num_processes:", accelerator.num_processes)
    print("local_rank:", accelerator.local_process_index)
    print("device:", accelerator.device)

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # Import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load text encoders
    text_encoder_one, text_encoder_two = load_text_encoders(args, text_encoder_cls_one, text_encoder_cls_two)
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    
    # Load transformer (only for time_text_embed to generate temb)
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # Freeze all pretrained models
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # Create token autoencoder
    token_upsampler = TokenUpsampler(
        in_channels=16,  # VAE latent channels
        feature_channels=args.tokenAE_feature_channels, 
        embedding_dim=transformer.inner_dim, 
        num_res_blocks=args.upsampler_num_res_blocks, 
        learn_residual=args.upsampler_learn_residual, 
        use_ada_norm=args.use_ada_norm,
        cnn_upsample_mode=args.cnn_upsample_mode,
        naive_upsample_mode=args.naive_upsample_mode,
    )
    token_downsampler = TokenDownsampler(
        in_channels=16,  # VAE latent channels
        feature_channels=args.tokenAE_feature_channels, 
        embedding_dim=transformer.inner_dim, 
        num_res_blocks=args.downsampler_num_res_blocks,
        learn_residual=args.downsampler_learn_residual,
        use_ada_norm=args.use_ada_norm,
        naive_downsample_mode=args.naive_downsample_mode,
    )
    
    # Create wrapper
    token_ae = TokenAEWrapper(token_upsampler, token_downsampler)

    # Log number of trainable parameters
    total_params = sum(p.numel() for p in token_ae.parameters())
    trainable_params = sum(p.numel() for p in token_ae.parameters() if p.requires_grad)
    logger.info(f"Token AE total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Token AE trainable parameters: {trainable_params / 1e6:.2f}M")

    # Mixed precision dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # Move frozen models to device
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    
    # Move token AE to device
    token_ae.to(accelerator.device, dtype=weight_dtype)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Custom saving & loading hooks
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), TokenAEWrapper):
                    torch.save(unwrap_model(model).token_upsampler.state_dict(), os.path.join(output_dir, "token_upsampler.pt"))
                    torch.save(unwrap_model(model).token_downsampler.state_dict(), os.path.join(output_dir, "token_downsampler.pt"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            model = models.pop()

            if isinstance(unwrap_model(model), TokenAEWrapper):
                upsampler_path = os.path.join(input_dir, "token_upsampler.pt")
                if os.path.exists(upsampler_path):
                    state_dict = torch.load(upsampler_path, map_location="cpu")
                    unwrap_model(model).token_upsampler.load_state_dict(state_dict)
                    logger.info(f"Loaded TokenUpsampler from {upsampler_path}")

                downsampler_path = os.path.join(input_dir, "token_downsampler.pt")
                if os.path.exists(downsampler_path):
                    state_dict = torch.load(downsampler_path, map_location="cpu")
                    unwrap_model(model).token_downsampler.load_state_dict(state_dict)
                    logger.info(f"Loaded TokenDownsampler from {downsampler_path}")
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimizer
    params_to_optimize = [{"params": token_ae.parameters(), "lr": args.learning_rate}]

    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders
    train_dataset = CustomDataset(
        data_root=args.instance_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
        debug_mode=args.debug_mode
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    # Scheduler and math around the number of training steps
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare with accelerator
    token_ae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        token_ae, optimizer, train_dataloader, lr_scheduler
    )

    # Recalculate total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers
    if accelerator.is_main_process:
        tracker_name = "token-ae"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running Token AE training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        token_ae.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(token_ae):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                prompts = batch["prompts"]

                # Get prompt embeddings for time conditioning
                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                    prompts, text_encoders, tokenizers
                )

                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.mode()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)

                bsz = pixel_values.shape[0]

                # Create dummy timestep (use 0 for autoencoder training)
                timesteps = torch.zeros(bsz, device=accelerator.device)
                    
                # Generate guidance embedding
                if transformer.config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(bsz)
                else:
                    guidance = None

                # Get time-text embedding for conditioning the token AE
                with torch.no_grad():
                    temb = transformer.time_text_embed(
                        timesteps.to(dtype=pooled_prompt_embeds.dtype), 
                        guidance.to(dtype=pooled_prompt_embeds.dtype), 
                        pooled_prompt_embeds
                    )

                # Forward pass through token AE
                decoded_latents, encoded_latents = token_ae(latents, temb)
                
                # L2 loss on latents
                loss = ((decoded_latents - latents).float() ** 2).mean()

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(token_ae.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging and checkpointing
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    # Save checkpoint
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # Log images
                    if global_step % args.log_image_steps == 0:
                        # Decode latents back to images for visualization
                        with torch.no_grad():
                            # Decode original latents
                            original_decoded = (latents / vae.config.scaling_factor) + vae.config.shift_factor
                            original_images = vae.decode(original_decoded.to(vae.dtype), return_dict=False)[0]
                            
                            # Decode reconstructed latents
                            reconstructed_decoded = (decoded_latents / vae.config.scaling_factor) + vae.config.shift_factor
                            reconstructed_images = vae.decode(reconstructed_decoded.to(vae.dtype), return_dict=False)[0]

                        images_dict = {
                            "original": tensor_to_pil_image(original_images.float()),
                            "reconstructed": tensor_to_pil_image(reconstructed_images.float()),
                            "latent_original": tensor_to_pil_latent(latents.float()),
                            "latent_reconstructed": tensor_to_pil_latent(decoded_latents.float()),
                            "latent_error": tensor_to_pil_latent((decoded_latents.float() - latents.float()) ** 2),
                        }
                        log_images(accelerator, images_dict)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    
    # Save final model
    if accelerator.is_main_process:
        unwrapped_token_ae = unwrap_model(token_ae)
        
        # Save final checkpoint
        final_save_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_save_path, exist_ok=True)
        torch.save(unwrapped_token_ae.token_upsampler.state_dict(), os.path.join(final_save_path, "token_upsampler.pt"))
        torch.save(unwrapped_token_ae.token_downsampler.state_dict(), os.path.join(final_save_path, "token_downsampler.pt"))
        logger.info(f"Saved final model to {final_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
