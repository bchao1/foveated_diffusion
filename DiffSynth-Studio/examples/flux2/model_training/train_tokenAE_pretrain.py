"""
TokenAE pretraining script.

Trains token upsampler and downsampler to compress/decompress VAE latents:
  image -> VAE encode -> down -> up -> VAE decode -> reconstructed image
  Loss = MSE(reconstructed, original)

Uses same dataset and image processing as train_foveated.py.
VAE is frozen; only token_upsampler and token_downsampler are trained.
"""

import torch
import sys
import os
import argparse
import yaml
import accelerate
from einops import rearrange
from PIL import Image
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../image_generation"))

from PIL import ImageFile, PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 1000 * (1024**2)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.flux2_image_foveated import Flux2FoveatedImagePipeline
from diffsynth.pipelines.flux2_image import ModelConfig
from diffsynth.diffusion import add_general_config, add_image_size_config, ModelLogger

try:
    from adapter_foveation_time_cond import TokenDownsampler, TokenUpsampler
except ImportError:
    TokenUpsampler = None
    TokenDownsampler = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _token_ae_config_from_args(args):
    """Build token AE config dict for saving."""
    return {
        "in_channels": getattr(args, "in_channels", 32),
        "feature_channels": getattr(args, "feature_channels", 64),
        "embedding_dim": getattr(args, "embedding_dim", 6144),
        "upsampler_num_res_blocks": getattr(args, "upsampler_num_res_blocks", 10),
        "downsampler_num_res_blocks": getattr(args, "downsampler_num_res_blocks", 1),
        "tokenAE_dont_use_learned_components": getattr(args, "tokenAE_dont_use_learned_components", False),
        "upsampler_normalization_type": getattr(args, "upsampler_normalization_type", "gn"),
        "downsampler_normalization_type": getattr(args, "downsampler_normalization_type", "none"),
        "upsampler_variance_correction": getattr(args, "upsampler_variance_correction", 1.0),
        "downsampler_variance_correction": getattr(args, "downsampler_variance_correction", 1.0),
        "naive_upsample_mode": getattr(args, "naive_upsample_mode", "nearest"),
        "naive_downsample_mode": getattr(args, "naive_downsample_mode", "nearest"),
    }


def load_token_ae(args, device):
    """Load TokenUpsampler and TokenDownsampler."""
    if TokenUpsampler is None or TokenDownsampler is None:
        raise ImportError("adapter_foveation_time_cond not found. Add image_generation to path.")
    in_ch = getattr(args, "in_channels", 32)
    feat_ch = getattr(args, "feature_channels", 64)
    emb_dim = getattr(args, "embedding_dim", 6144)
    token_upsampler = TokenUpsampler(
        in_channels=in_ch,
        feature_channels=feat_ch,
        embedding_dim=emb_dim,
        num_res_blocks=args.upsampler_num_res_blocks,
        naive_upsample_mode=args.naive_upsample_mode,
        learn_residual=True,
        use_learned_components=not args.tokenAE_dont_use_learned_components,
        normalization_type=args.upsampler_normalization_type,
        variance_correction=args.upsampler_variance_correction,
    )
    token_downsampler = TokenDownsampler(
        in_channels=in_ch,
        feature_channels=feat_ch,
        embedding_dim=emb_dim,
        num_res_blocks=args.downsampler_num_res_blocks,
        naive_downsample_mode=args.naive_downsample_mode,
        learn_residual=True,
        use_learned_components=not args.tokenAE_dont_use_learned_components,
        normalization_type=args.downsampler_normalization_type,
        variance_correction=args.downsampler_variance_correction,
    )
    if args.tokenAE_folder is not None:
        token_upsampler.load_state_dict(torch.load(os.path.join(args.tokenAE_folder, "token_upsampler.pt"), map_location="cpu"))
        token_downsampler.load_state_dict(torch.load(os.path.join(args.tokenAE_folder, "token_downsampler.pt"), map_location="cpu"))
    return token_upsampler.to(device).to(torch.bfloat16), token_downsampler.to(device).to(torch.bfloat16)


class TokenAEPretrainModule(torch.nn.Module):
    """Module for TokenAE pretraining: image -> encode -> down -> up -> decode -> MSE loss."""

    def __init__(self, pipe, token_upsampler, token_downsampler, token_ae_config=None):
        super().__init__()
        self.pipe = pipe
        self.token_upsampler = token_upsampler
        self.token_downsampler = token_downsampler
        self.token_ae_config = token_ae_config or {}
        # Freeze pipeline (VAE, etc.)
        self.pipe.eval()
        self.pipe.requires_grad_(False)

    def forward(self, data):
        """
        data: dict with "image" (PIL) and optionally "height", "width"
        Returns: loss (MSE between original and reconstructed)
        """
        image = data["image"]
        height = data.get("height", image.size[1])
        width = data.get("width", image.size[0])
        device = self.pipe.device
        dtype = self.pipe.torch_dtype

        self.pipe.load_models_to_device(["vae"])
        # Preprocess: PIL -> [1, C, H, W], range [-1, 1]
        image_tensor = self.pipe.preprocess_image(image)
        image_tensor = image_tensor.to(device=device, dtype=dtype)

        # Encode
        latents = self.pipe.vae.encode(image_tensor)  # [B, 128, h, w]
        # Unpack to spatial for token modules: [B, 128, h, w] -> [B, 32, 2h, 2w]
        latents_spatial = rearrange(latents, "B (C P Q) h w -> B C (h P) (w Q)", P=2, Q=2, C=32)

        # Dummy temb (zeros) for gn/none normalization
        B = latents_spatial.shape[0]
        temb = torch.zeros(B, 6144, device=device, dtype=dtype)

        # Down -> Up
        latents_down = self.token_downsampler(latents_spatial, temb=temb, foveation_mask=None)
        latents_up = self.token_upsampler(latents_down, temb=temb, foveation_mask=None)

        # Pack for VAE decode: [B, 32, 2h, 2w] -> [B, 128, h, w]
        latents_packed = rearrange(latents_up, "B C (h P) (w Q) -> B (C P Q) h w", P=2, Q=2, C=32)

        # Decode
        reconstructed = self.pipe.vae.decode(latents_packed)
        self.pipe.load_models_to_device([])

        loss = torch.nn.functional.mse_loss(reconstructed.float(), image_tensor.float())
        return loss


class TokenAEPretrainLogger(ModelLogger):
    """Logger for TokenAE pretrain: saves checkpoints, logs original + reconstructed to WandB."""

    def __init__(
        self,
        output_path: str,
        use_wandb: bool = True,
        project_name: str = "tokenAE-pretrain",
        run_name: str = None,
        config: dict = None,
        log_steps: int = 500,
    ):
        super().__init__(output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.log_steps = log_steps
        self._wandb_initialized = False

    def init_wandb(self, accelerator):
        if not self.use_wandb or self._wandb_initialized:
            return
        if accelerator.is_main_process:
            wandb.init(project=self.project_name, name=self.run_name, config=self.config, resume="allow")
            self._wandb_initialized = True

    def log_images(self, model, data, accelerator):
        """Run one batch and log original + reconstructed images to WandB."""
        if not self._wandb_initialized or not accelerator.is_main_process:
            return
        unwrapped = accelerator.unwrap_model(model)
        with torch.no_grad():
            image = data["image"]
            device = unwrapped.pipe.device
            dtype = unwrapped.pipe.torch_dtype
            unwrapped.pipe.load_models_to_device(["vae"])
            image_tensor = unwrapped.pipe.preprocess_image(image)
            image_tensor = image_tensor.to(device=device, dtype=dtype)
            latents = unwrapped.pipe.vae.encode(image_tensor)
            latents_spatial = rearrange(latents, "B (C P Q) h w -> B C (h P) (w Q)", P=2, Q=2, C=32)
            B = latents_spatial.shape[0]
            temb = torch.zeros(B, 6144, device=device, dtype=dtype)
            latents_down = unwrapped.token_downsampler(latents_spatial, temb=temb, foveation_mask=None)
            latents_up = unwrapped.token_upsampler(latents_down, temb=temb, foveation_mask=None)
            latents_packed = rearrange(latents_up, "B C (h P) (w Q) -> B (C P Q) h w", P=2, Q=2, C=32)
            reconstructed = unwrapped.pipe.vae.decode(latents_packed)
            unwrapped.pipe.load_models_to_device([])
            # Tensor -> PIL
            orig_pil = unwrapped.pipe.vae_output_to_image(image_tensor)
            recon_pil = unwrapped.pipe.vae_output_to_image(reconstructed)
        wandb_images = [
            wandb.Image(orig_pil, caption="original"),
            wandb.Image(recon_pil, caption="reconstructed"),
        ]
        wandb.log({"tokenAE/original": wandb_images[0], "tokenAE/reconstructed": wandb_images[1]}, step=self.num_steps)

    def save_token_ae(self, model, accelerator):
        if not accelerator.is_main_process:
            return
        unwrapped = accelerator.unwrap_model(model)
        path = os.path.join(self.output_path, "tokenAE", f"step-{self.num_steps}")
        os.makedirs(path, exist_ok=True)
        torch.save(unwrapped.token_upsampler.state_dict(), os.path.join(path, "token_upsampler.pt"))
        torch.save(unwrapped.token_downsampler.state_dict(), os.path.join(path, "token_downsampler.pt"))
        if unwrapped.token_ae_config:
            cfg = dict(unwrapped.token_ae_config)
            up = unwrapped.token_upsampler
            if getattr(up, "variance_correction", None) is not None and hasattr(up.variance_correction, "data"):
                cfg["upsampler_variance_correction"] = up.variance_correction.detach().float().item()
            down = unwrapped.token_downsampler
            if getattr(down, "variance_correction", None) is not None and hasattr(down.variance_correction, "data"):
                cfg["downsampler_variance_correction"] = down.variance_correction.detach().float().item()
            with open(os.path.join(path, "tokenAE_config.yaml"), "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _log_variance_correction(self, accelerator, model):
        if not self._wandb_initialized or not accelerator.is_main_process:
            return
        unwrapped = accelerator.unwrap_model(model)
        metrics = {}
        for name, module in [("upsampler", unwrapped.token_upsampler), ("downsampler", unwrapped.token_downsampler)]:
            if getattr(module, "variance_correction", None) is not None and hasattr(module.variance_correction, "data"):
                metrics[f"tokenAE/variance_correction_{name}"] = module.variance_correction.detach().float().item()
        if metrics:
            wandb.log(metrics, step=self.num_steps)

    def on_training_start(self, accelerator, model, dataloader):
        """Log initial images at step 0."""
        self.init_wandb(accelerator)
        if accelerator.is_main_process and self.log_steps > 0:
            for data in dataloader:
                self.log_images(model, data, accelerator)
                break

    def on_step_end(self, accelerator, model, save_steps=None, loss=None, data=None, **kwargs):
        self.num_steps += 1
        self.init_wandb(accelerator)
        if loss is not None and accelerator.is_main_process and self._wandb_initialized:
            wandb.log({"tokenAE/loss": float(loss)}, step=self.num_steps)
        if accelerator.is_main_process and self._wandb_initialized:
            self._log_variance_correction(accelerator, model)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_token_ae(model, accelerator)
        if self.log_steps > 0 and self.num_steps % self.log_steps == 0 and data is not None:
            self.log_images(model, data, accelerator)

    def on_epoch_end(self, accelerator, model, epoch_id):
        self.save_token_ae(model, accelerator)

    def on_training_end(self, accelerator, model, save_steps=None):
        self.save_token_ae(model, accelerator)
        if accelerator.is_main_process and self._wandb_initialized:
            wandb.finish()


def launch_tokenAE_pretrain(accelerator, dataset, model, model_logger, args):
    optimizer = torch.optim.AdamW(
        list(model.token_upsampler.parameters()) + list(model.token_downsampler.parameters()),
        lr=args.tokenAE_learning_rate,
        weight_decay=args.weight_decay,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=args.dataset_num_workers
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    from tqdm import tqdm

    model_logger.on_training_start(accelerator, model, dataloader)

    for epoch_id in range(args.num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, args.save_steps, loss=loss, data=data)
        model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, args.save_steps)


def tokenAE_pretrain_parser():
    parser = argparse.ArgumentParser(description="TokenAE pretraining")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenAE_folder", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--tokenAE_learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="tokenAE-pretrain")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_steps", type=int, default=500, help="Log original+reconstructed to WandB every N steps")
    # Token AE args (same as train_foveated)
    parser.add_argument("--in_channels", type=int, default=32)
    parser.add_argument("--feature_channels", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=6144)
    parser.add_argument("--upsampler_num_res_blocks", type=int, default=10)
    parser.add_argument("--downsampler_num_res_blocks", type=int, default=1)
    parser.add_argument("--tokenAE_dont_use_learned_components", action="store_true")
    parser.add_argument("--upsampler_normalization_type", type=str, default="gn", choices=["ada", "gn", "none"])
    parser.add_argument("--downsampler_normalization_type", type=str, default="none", choices=["ada", "gn", "none"])
    parser.add_argument("--upsampler_variance_correction", type=float, default=1.0)
    parser.add_argument("--downsampler_variance_correction", type=float, default=1.0)
    parser.add_argument("--naive_upsample_mode", type=str, default="nearest", choices=["nearest", "bilinear"])
    parser.add_argument("--naive_downsample_mode", type=str, default="nearest", choices=["nearest", "bilinear"])
    return parser


if __name__ == "__main__":
    parser = tokenAE_pretrain_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
    )

    # Load pipeline (same as train_foveated)
    model_configs = []
    if args.model_paths:
        import json
        for path in json.loads(args.model_paths):
            model_configs.append(ModelConfig(path=path))
    if args.model_id_with_origin_paths:
        for pair in args.model_id_with_origin_paths.split(","):
            pair = pair.strip()
            idx = pair.rfind(":")
            if idx >= 0:
                model_configs.append(ModelConfig(model_id=pair[:idx].strip(), origin_file_pattern=pair[idx + 1 :].strip()))
    if not model_configs:
        model_configs = [
            ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ]
    tokenizer_config = ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="tokenizer/") if args.tokenizer_path else None
    if args.tokenizer_path and ":" in args.tokenizer_path:
        mid, pat = args.tokenizer_path.split(":")
        tokenizer_config = ModelConfig(model_id=mid.strip(), origin_file_pattern=pat.strip())

    pipe = Flux2FoveatedImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=accelerator.device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    token_upsampler, token_downsampler = load_token_ae(args, accelerator.device)
    token_ae_config = _token_ae_config_from_args(args)

    model = TokenAEPretrainModule(pipe, token_upsampler, token_downsampler, token_ae_config)

    model_logger = TokenAEPretrainLogger(
        output_path=args.output_path,
        use_wandb=args.use_wandb,
        project_name=args.wandb_project,
        run_name=args.wandb_run_name,
        config=vars(args),
        log_steps=args.log_steps,
    )
    launch_tokenAE_pretrain(accelerator, dataset, model, model_logger, args)
