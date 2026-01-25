
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
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image, UnidentifiedImageError, PngImagePlugin, ImageFile
from diffusers.utils.torch_utils import randn_tensor

from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast
# use CLIPTextModel instead of CLIPTextModelWithProjection to match inference pipeline
# https://www.reddit.com/r/localdiffusion/comments/1995s27/difference_between_transformers_cliptextmodel_and/

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)

from transformer_flux_crpa import FluxPhaseAlignedTransformer2DModel 
from pipeline_flux_foveation_tokenAE import FluxPipeline_Foveation
from utils import create_foveation_mask

from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

# for token upsampling and downsampling. either time-conditioned or no

from adapter_foveation_time_cond import TokenDownsampler, TokenUpsampler
#from adapter_foveation import TokenDownsampler, TokenUpsampler

# handle image loading
PngImagePlugin.MAX_TEXT_CHUNK = 1000 * (1024**2)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0")

logger = get_logger(__name__)
logging.getLogger("transformer_flux_crpa").setLevel(logging.ERROR)

if is_torch_npu_available():
    import torch_npu

    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)




def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def log_validation(
    pipeline,
    foveation_mask,
    token_upsampler,
    token_downsampler,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    # ===============================
    token_upsampler = token_upsampler.to(accelerator.device) # 0101: tokenAE
    token_downsampler = token_downsampler.to(accelerator.device) # 0101: tokenAE
    # ===============================
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    autocast_ctx = nullcontext()

    # generate masks
    foveation_mask_list = []
    #r_list = np.linspace(0.1, 1.0, args.num_validation_images)
    r_list = [0.5]
    for r in r_list:
        foveation_mask = create_foveation_mask(args.resolution // 16, args.resolution // 16, center=(0, 0), r = r)
        foveation_mask = foveation_mask.to(accelerator.device, dtype=torch_dtype)
        foveation_mask_list.append(foveation_mask)

    with autocast_ctx:
        images = [pipeline(**pipeline_args, foveation_mask=foveation_mask_list[i], token_upsampler=token_upsampler, token_downsampler=token_downsampler, generator=generator).images[0] for i in range(len(foveation_mask_list))]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_torch_npu_available():
        torch_npu.npu.empty_cache()

    return images

def tensor_to_pil(flow_tensor):
    """
    Converts a flow tensor [B, C, H, W] to a list of PIL Images.
    Visualizes the first 3 channels mapped to RGB.
    """
    # 1. Detach and move to CPU
    # Take first 3 channels for RGB visualization
    # Shape: [B, 3, H, W]
    viz = flow_tensor[:, :3, :, :].detach().float().cpu()

    # 2. Normalize to [0, 1] range for visualization
    # We normalize each image individually based on its max absolute value
    # This ensures 0 flow looks "Gray" (0.5), positive is bright, negative is dark.
    images = []
    for i in range(viz.shape[0]):
        max_val = torch.max(torch.abs(viz[i]))
        if max_val > 1e-6:
            # Scale to [-0.5, 0.5] then shift to [0, 1]
            # Result: 0.0 -> 0.5 (Gray), +Max -> 1.0 (White), -Max -> 0.0 (Black)
            img_tensor = (viz[i] / (2 * max_val)) + 0.5
        else:
            img_tensor = torch.full_like(viz[i], 0.5) # Gray if zero flow

        # 3. Clamp and convert to uint8 [H, W, C]
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # 4. Convert to PIL
        images.append(Image.fromarray(img_np))
        
    return images
    
def log_images(accelerator, images_dict, is_final_validation=False):

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"

        log_payload = {}

        for image_name, image_list in images_dict.items():
            # 2. Create a unique key using the phase and the image type
            # Example keys: "validation/target_flow", "validation/pred_flow"
            key = f"{phase_name}/{image_name}"
            
            # 3. Add to the payload
            log_payload[key] = [
                wandb.Image(image, caption=f"{image_name} {i}") 
                for i, image in enumerate(image_list)
            ]
        
        # 4. Log everything at once
        # This guarantees they all appear on the same timeline step
        if tracker.name == "wandb":
            tracker.log(log_payload)
        else:
            raise ValueError(f"Tracker {tracker.name} not found!")
        

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
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
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
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run validation every X epochs. validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-foveation",
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
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
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
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
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
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
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
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

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
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
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
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[
            f.lower() for f in dir(transforms.InterpolationMode) if not f.startswith("__") and not f.endswith("__")
        ],
        help="The image interpolation method to use for resizing images.",
    )
    parser.add_argument("--enable_npu_flash_attention", action="store_true", help="Enabla Flash Attention for NPU")
    parser.add_argument("--debug_mode", default=None, type=str)

    # token AE params
    parser.add_argument("--downsampler_variance_correction", action="store_true")
    parser.add_argument("--upsampler_variance_correction", action="store_true")
    parser.add_argument("--downsampler_learn_residual", action="store_true")
    parser.add_argument("--upsampler_learn_residual", action="store_true")
    parser.add_argument("--use_ada_norm", action="store_true")
    parser.add_argument("--tokenAE_feature_channels", type=int, default=16)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class CustomDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        data_root,
        size=1024,
        repeats=1,
        center_crop=False,
        debug_mode=None
    ):
        self.data_root = data_root
        self.size = size
        self.center_crop = center_crop


        self.filenames = []
        self.prompts = []

        from torchvision import transforms
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

            json_list.sort() # important!
            for json_str in json_list:
                example = json.loads(json_str)
                self.filenames.append(example["file_name"])
                self.prompts.append(example["text"])

        if debug_mode == "single":
            self.filenames = [self.filenames[0]] * len(json_list)
            self.prompts = [self.prompts[0]] * len(json_list)
            print(self.prompts[0])
                

        self._length = len(self.filenames)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        for _ in range(5):  # retry a few times
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
                # corrupted image → sample a different one
                index = random.randint(0, self._length - 1)


def collate_fn(examples):
    pixel_values = [example["pixel_values"] for example in examples]
    prompts = [example["prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    return batch



def tokenize_prompt(tokenizer, prompt, max_sequence_length):
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
    return text_input_ids


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

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
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
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
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
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
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

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

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

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxPhaseAlignedTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    transformer.requires_grad_(True) # 0101: don't train the transformer
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # 0101: tokenAE 
    # ===============================
    token_upsampler = TokenUpsampler(in_channels=16, # VAE channels
                                     feature_channels=args.tokenAE_feature_channels, 
                                     embedding_dim=transformer.inner_dim, 
                                     num_res_blocks=10, 
                                     learn_residual=args.upsampler_learn_residual, 
                                     use_ada_norm=args.use_ada_norm
                                    )
    token_downsampler = TokenDownsampler(in_channels=16,
                                         feature_channels=args.tokenAE_feature_channels, 
                                         embedding_dim=transformer.inner_dim, 
                                         num_res_blocks=10,
                                         learn_residual=args.downsampler_learn_residual,
                                         use_ada_norm=args.use_ada_norm
                                        )
    # ===============================

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            transformer.set_attention_backend("_native_npu")
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu device ")

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    
    # move tokenAE to device and float32 (CNN use float32, so explicitly cast when using)
    # ===============================
    token_upsampler.to(accelerator.device, dtype=weight_dtype)
    token_downsampler.to(accelerator.device, dtype=weight_dtype)
    # ===============================

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # 0101: TODO: no need to save transformer and text encoders, only tokenAE needs to be saved
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), FluxPhaseAlignedTransformer2DModel):
                    #unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"))
                    pass
                elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                    #if isinstance(unwrap_model(model), CLIPTextModelWithProjection):
                    #    unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder"))
                    #else:
                    #    unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_2"))
                    pass
                # ===============================
                elif isinstance(unwrap_model(model), (TokenUpsampler, TokenDownsampler)):
                    if isinstance(unwrap_model(model), TokenUpsampler):
                        #unwrap_model(model).save_pretrained(os.path.join(output_dir, "token_upsampler"))
                        torch.save(unwrap_model(model).state_dict(), os.path.join(output_dir, "token_upsampler.pt"))
                    else:
                        #unwrap_model(model).save_pretrained(os.path.join(output_dir, "token_downsampler"))
                        torch.save(unwrap_model(model).state_dict(), os.path.join(output_dir, "token_downsampler.pt"))
                # ===============================
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(unwrap_model(model), FluxPhaseAlignedTransformer2DModel):
                #load_model = FluxPhaseAlignedTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                #model.register_to_config(**load_model.config)

                #model.load_state_dict(load_model.state_dict())
                pass
            elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                try:
                    #load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder")
                    #model(**load_model.config)
                    #model.load_state_dict(load_model.state_dict())
                    pass
                except Exception:
                    try:
                        #load_model = T5EncoderModel.from_pretrained(input_dir, subfolder="text_encoder_2")
                        #model(**load_model.config)
                        #model.load_state_dict(load_model.state_dict())
                        pass
                    except Exception:
                        raise ValueError(f"Couldn't load the model of type: ({type(model)}).")
            # ===============================
            elif isinstance(unwrap_model(model), (TokenUpsampler, TokenDownsampler)):
                if isinstance(unwrap_model(model), TokenUpsampler):
                    #load_model = TokenUpsampler.from_pretrained(input_dir, subfolder="token_upsampler")
                    load_model = torch.load(os.path.join(input_dir, "token_upsampler.pt"))
                else:
                    #load_model = TokenDownsampler.from_pretrained(input_dir, subfolder="token_downsampler")
                    load_model = torch.load(os.path.join(input_dir, "token_downsampler.pt"))
                #model(**load_model.config)
                model.load_state_dict(load_model.state_dict())
            # ===============================
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer.parameters(), "lr": args.learning_rate} # 0101: don't train the transformer
    # ===============================
    token_AE_parameters_with_lr = {
        "params":  [
            *token_upsampler.parameters(), 
            *token_downsampler.parameters()
        ], 
        "lr": args.learning_rate
    } # 0101: tokenAE
    # ===============================
    params_to_optimize = [token_AE_parameters_with_lr, transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
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

    # Dataset and DataLoaders creation:
    train_dataset = CustomDataset(
        data_root=args.instance_data_dir,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        debug_mode=args.debug_mode
    )
    if args.debug_mode == "single":
        args.validation_prompt = train_dataset.prompts[0]

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


    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
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

    # Prepare everything with our `accelerator`.
    # 0101: add token_upsampler and token_downsampler to the list of models to prepare
    transformer, token_upsampler, token_downsampler, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, token_upsampler, token_downsampler, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "flux-foveation"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
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
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # for fixed seed generation
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    fixed_noise = None
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        token_upsampler.train()
        token_downsampler.train()

        for step, batch in enumerate(train_dataloader):
            # 0101: prepare foveation masks
            # ===============================
            r = np.random.choice(np.linspace(0.1, 1.0, 10))
            r = 0.5
            
            center_min = -(1 - r) * 0.5
            center_max = (1 - r) * 0.5
            center_x, center_y = np.random.uniform(center_min, center_max, 2)
            center_x, center_y = 0, 0
            
            foveation_mask = create_foveation_mask(args.resolution // 16, args.resolution // 16, center=(center_x, center_y), r = r)
            foveation_mask = foveation_mask.to(accelerator.device, dtype=weight_dtype)
            joint_attention_kwargs = {}
            # ===============================


            models_to_accumulate = [transformer, token_upsampler, token_downsampler] # 0101: do not learn transformer
            #models_to_accumulate = [token_upsampler, token_downsampler] # 0101: learn tokenAE
            
            with accelerator.accumulate(models_to_accumulate):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                prompts = batch["prompts"]

                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                    prompts, text_encoders, tokenizers
                )

                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

                bsz, latent_height, latent_width = model_input.shape[0], model_input.shape[2] // 2, model_input.shape[3] // 2
                #print("train", model_input.shape, latent_height, foveation_mask.shape)
                #exit()
                latent_image_ids, resolution_mask, resolution_mask_top_left = FluxPipeline_Foveation._prepare_latent_image_ids(
                    bsz, latent_height, latent_width,
                    accelerator.device,
                    weight_dtype,
                    foveation_mask=foveation_mask
                )
                joint_attention_kwargs.update({"resolution_mask": resolution_mask, "resolution_mask_top_left": resolution_mask_top_left})

                
                # Sample noise that we'll add to the latents
                if args.debug_mode == "single":
                    if fixed_noise is None:
                        print("setting fixed noise...")
                        fixed_noise = randn_tensor(model_input.shape, generator=generator, device=model_input.device, dtype=model_input.dtype) 
                    #print(noise)
                    noise = fixed_noise
                else:
                    noise = torch.randn_like(model_input)
            
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                
                #if args.debug_mode == "single":
                #    # fix timestep training
                #    timesteps = torch.full_like(timesteps, 500)
                
                # handle guidance
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                # generate time and text embeddings (for token samplers)
                #print(timesteps.dtype, guidance.dtype, pooled_prompt_embeds.dtype)
                #exit()
                temb = accelerator.unwrap_model(transformer).time_text_embed(timesteps.to(dtype=pooled_prompt_embeds.dtype), guidance.to(dtype=pooled_prompt_embeds.dtype), pooled_prompt_embeds)
                #temb = None

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline_Foveation._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                packed_noisy_model_input = FluxPipeline_Foveation._downsample_latents(
                    packed_noisy_model_input, latent_height, latent_width, 
                    foveation_mask=foveation_mask, token_downsampler=token_downsampler,
                    temb=temb
                )

                # Predict the noise residual
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
                )[0]
                # upscaling height & width as discussed in https://github.com/huggingface/diffusers/pull/9257#discussion_r1731108042
                """
                model_pred = FluxPipeline_Foveation._upsample_noise_pred(
                    model_pred, latent_height, latent_width, 
                    foveation_mask=foveation_mask, token_upsampler=token_upsampler,
                    temb=temb
                )
                model_pred = FluxPipeline_Foveation._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                """
                model_pred = FluxPipeline_Foveation._upsample_noise_pred_unpacked(
                    model_pred, 
                    latent_height, 
                    latent_width, 
                    vae_scale_factor=vae_scale_factor,
                    foveation_mask=foveation_mask, 
                    token_upsampler=token_upsampler, 
                    temb=temb
                ) # Returns [B, 16, 128, 128] directly

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input


                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(transformer.parameters(), token_upsampler.parameters(), token_downsampler.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
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

                    if global_step % 100 == 0:
                        # log images
                        images_dict = {
                            "model_pred": tensor_to_pil(model_pred.float()),
                            "target": tensor_to_pil(target.float()),
                            "error": tensor_to_pil((model_pred.float() - target.float())**2)
                        }
                        log_images(accelerator, images_dict)

                    if (args.validation_prompt is not None) and (global_step % 1000 == 0):
                        # create pipeline
                        text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
                        text_encoder_one.to(weight_dtype)
                        text_encoder_two.to(weight_dtype)
                        
                        pipeline = FluxPipeline_Foveation.from_pretrained(
                            args.pretrained_model_name_or_path,
                            vae=vae,
                            text_encoder=unwrap_model(text_encoder_one),
                            text_encoder_2=unwrap_model(text_encoder_two),
                            transformer=unwrap_model(transformer),
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=weight_dtype,
                        )
                        pipeline_args = {
                            "prompt": args.validation_prompt, 
                            "guidance_scale": 3.5, 
                            "height": 1024, 
                            "width": 1024, 
                            "max_sequence_length": 256,
                            "foveation_start_step": 0
                        }
                        r = np.random.choice(np.linspace(0.1, 1.0, 10))
                        foveation_mask = create_foveation_mask(args.resolution // 16, args.resolution // 16, center=(0, 0), r = r)
                        foveation_mask = foveation_mask.to(accelerator.device, dtype=weight_dtype)
                        images = log_validation(
                            pipeline=pipeline,
                            foveation_mask=foveation_mask,
                            token_upsampler=token_upsampler,
                            token_downsampler=token_downsampler,
                            args=args,
                            accelerator=accelerator,
                            pipeline_args=pipeline_args,
                            epoch=epoch,
                            torch_dtype=weight_dtype
                        )
                        
                        del text_encoder_one, text_encoder_two
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif is_torch_npu_available():
                            torch_npu.npu.empty_cache()
                        gc.collect()
        
                        images = None
                        del pipeline

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
                

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)

        pipeline = FluxPipeline_Foveation.from_pretrained(args.pretrained_model_name_or_path, transformer=transformer)

        # save the pipeline
        pipeline.save_pretrained(args.output_dir)

        # Final inference
        # Load previous pipeline
        pipeline = FluxPipeline_Foveation.from_pretrained(
            args.output_dir,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {
                    "prompt": args.validation_prompt, 
                    "guidance_scale": 3.5, 
                    "height": 1024, 
                    "width": 1024, 
                    "max_sequence_length": 256,
                    "foveation_start_step": 0
                }
            r = np.random.choice(np.linspace(0.1, 1.0, 10))
            foveation_mask = create_foveation_mask(args.resolution // 16, args.resolution // 16, center=(0, 0), r = r)
            foveation_mask = foveation_mask.to(accelerator.device, dtype=weight_dtype)
            images = log_validation(
                pipeline=pipeline,
                foveation_mask=foveation_mask,
                token_upsampler=token_upsampler,
                token_downsampler=token_downsampler,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                epoch=epoch,
                is_final_validation=True,
                torch_dtype=weight_dtype
            )


        images = None
        del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
