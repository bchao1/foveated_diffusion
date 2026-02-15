import torch, sys, os, argparse, accelerate
import yaml
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.flux2_image_foveated import Flux2FoveatedImagePipeline
from diffsynth.pipelines.flux2_image import ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../image_generation"))

# handling load large images
from PIL import Image, ImageFile, PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 1000 * (1024**2)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from adapter_foveation_time_cond import TokenDownsampler, TokenUpsampler
except ImportError:
    TokenUpsampler = None
    TokenDownsampler = None

try:
    from diffsynth.models.flux2_token_ae import Flux2TokenDownsampler, Flux2TokenUpsampler
except ImportError:
    Flux2TokenDownsampler = None
    Flux2TokenUpsampler = None

# Keys to save for tokenAE config (all archs, matches train_tokenAE_pretrain.py)
TOKEN_AE_CONFIG_KEYS = {
    "tokenAE_arch",
    "in_channels",
    "feature_channels",
    "embedding_dim",
    "upsampler_num_res_blocks",
    "downsampler_num_res_blocks",
    "tokenAE_dont_use_learned_components",
    "upsampler_normalization_type",
    "downsampler_normalization_type",
    "upsampler_variance_correction",
    "downsampler_variance_correction",
    "naive_upsample_mode",
    "naive_downsample_mode",
    "flux2_naive_downsample_mode",
    "flux2_naive_upsample_mode",
    "flux2_resnet_eps",
    "flux2_resnet_act_fn",
    "flux2_resnet_groups",
    "flux2_downsampler_add_norm_act_before_conv_out",
    "flux2_upsampler_add_norm_act_before_conv_out",
    "flux2_upsampler_num_pre_upsampler_res_blocks",
    "flux2_downsampler_add_attention",
    "flux2_downsampler_attention_head_dim",
    "flux2_downsampler_attn_groups",
    "flux2_upsampler_add_attention",
    "flux2_upsampler_attention_head_dim",
    "flux2_upsampler_attn_groups",
}


def _token_ae_config_from_args(args):
    """Build token AE config dict for saving. Includes all arch-specific params for flexibility."""
    cfg = {"tokenAE_arch": getattr(args, "tokenAE_arch", "adapter")}
    for key in TOKEN_AE_CONFIG_KEYS:
        if key != "tokenAE_arch" and hasattr(args, key):
            cfg[key] = getattr(args, key)
    return cfg


def _load_token_ae_config_from_folder(folder: str) -> dict:
    """Load tokenAE_config.yaml from a checkpoint folder if present."""
    if folder is None:
        return {}
    for name in ("tokenAE_config.yaml", "tokenAE_config.yml"):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    cfg = yaml.safe_load(f)
                return cfg if isinstance(cfg, dict) else {}
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
                return {}
    return {}


class Flux2FoveatedImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        token_upsampler=None,
        token_downsampler=None,
        token_ae_config=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = self.parse_path_or_model_id(tokenizer_path, default_value=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="tokenizer/"))
        self.pipe = Flux2FoveatedImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # set token upsampler and downsampler (register as our children so they're in the module tree for optimizer/grads)
        self.pipe.token_upsampler = token_upsampler
        self.pipe.token_downsampler = token_downsampler
        self.token_ae_config = token_ae_config or {}

        # Build effective trainable models: trainable_models + lora_base_model + token AE (if present)
        parts = []
        if trainable_models:
            parts.extend(trainable_models.split(","))
        if lora_base_model is not None:
            parts.append(lora_base_model)
        if token_upsampler is not None and token_downsampler is not None:
            parts.extend(["token_upsampler", "token_downsampler"])
        effective_trainable = ",".join(parts) if parts else None
        self._main_trainable = [p for p in (trainable_models.split(",") if trainable_models else []) + ([lora_base_model] if lora_base_model else [])]
        self._token_ae_trainable = ["token_upsampler", "token_downsampler"] if (token_upsampler is not None and token_downsampler is not None) else []

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, effective_trainable,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FoveatedFlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi), # use foveated flow match sft loss
            "distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FoveatedDistillLoss(pipe, **inputs_shared, **inputs_posi), # use foveated distillation loss
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FoveatedFlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi), # use foveated flow match sft loss
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self._last_phase = None

    def set_training_phase(self, step: int, tokenAE_train_steps=None):
        """Switch from tokenAE-only to joint training. tokenAE_train_steps: train token AE for first N steps, then train both jointly."""
        if tokenAE_train_steps is None or not self._token_ae_trainable:
            print("<Flux2FoveatedImageTrainingModule> ===> No token AE training required")
            return
        if step < tokenAE_train_steps:
            phase = "tokenAE"
            if not self._last_phase == "tokenAE":
                print("<Flux2FoveatedImageTrainingModule> ===> Training token AE for the first {} steps".format(tokenAE_train_steps))
        else:
            phase = "joint"
            if not self._last_phase == "joint":
                print("<Flux2FoveatedImageTrainingModule> ===> Training joint model")
        if self._last_phase == phase:
            return
        self._last_phase = phase
        # set all modules to eval mode and no grad
        self.pipe.eval()
        self.pipe.requires_grad_(False)

        # selectively set modules to train mode based on training phase
        if phase == "tokenAE":
            for name in self._token_ae_trainable:
                module = self.pipe.get_module(self.pipe, name)
                if module is not None:
                    print("<Flux2FoveatedImageTrainingModule> ===> Setting {} to train mode".format(name))
                    module.train()
                    module.requires_grad_(True)
        else:
            for name in self._main_trainable + self._token_ae_trainable:
                module = self.pipe.get_module(self.pipe, name)
                if module is not None:
                    print("<Flux2FoveatedImageTrainingModule> ===> Setting {} to train mode".format(name))
                    module.train()
                    module.requires_grad_(True)
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "embedded_guidance": 1.0,
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        """Export trainable state dict, excluding token upsampler/downsampler (saved separately)."""
        state_dict = super().export_trainable_state_dict(state_dict, remove_prefix)
        # Exclude token AE params from main checkpoint; they are saved to tokenAE subdirectory
        excluded_prefixes = ("pipe.token_upsampler.", "pipe.token_downsampler.")
        return {k: v for k, v in state_dict.items() if not any(k.startswith(p) for p in excluded_prefixes)}

    def save_token_ae(self, path, step=None):
        """Save token upsampler and downsampler to a separate folder. If step is provided, save to path/step-{step}/."""
        if self.pipe.token_upsampler is None or self.pipe.token_downsampler is None:
            return
        if step is not None:
            path = os.path.join(path, f"step-{step}")
        os.makedirs(path, exist_ok=True)
        torch.save(self.pipe.token_upsampler.state_dict(), os.path.join(path, "token_upsampler.pt"))
        torch.save(self.pipe.token_downsampler.state_dict(), os.path.join(path, "token_downsampler.pt"))
        if self.token_ae_config:
            config_to_save = dict(self.token_ae_config)
            # Overwrite with current learned variance_correction from the models (nn.Parameters updated during training)
            up = self.pipe.token_upsampler
            if getattr(up, "variance_correction", None) is not None and hasattr(up.variance_correction, "data"):
                config_to_save["upsampler_variance_correction"] = up.variance_correction.detach().float().item()
            down = self.pipe.token_downsampler
            if getattr(down, "variance_correction", None) is not None and hasattr(down.variance_correction, "data"):
                config_to_save["downsampler_variance_correction"] = down.variance_correction.detach().float().item()
            with open(os.path.join(path, "tokenAE_config.yaml"), "w") as f:
                yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def load_token_ae(args, device):
    """Load TokenUpsampler and TokenDownsampler. Supports adapter and flux2 architectures.
    Uses tokenAE_config.yaml from tokenAE_folder when present; otherwise argparse defaults."""
    # When resuming, optionally override args from saved config
    if args.tokenAE_folder is not None:
        saved_cfg = _load_token_ae_config_from_folder(args.tokenAE_folder)
        if saved_cfg:
            for key in TOKEN_AE_CONFIG_KEYS:
                if key in saved_cfg:
                    setattr(args, key, saved_cfg[key])
    arch = getattr(args, "tokenAE_arch", "adapter")

    if arch == "flux2":
        if Flux2TokenDownsampler is None or Flux2TokenUpsampler is None:
            raise ImportError("diffsynth.models.flux2_token_ae not found. Install or set --tokenAE_arch adapter.")
        in_ch = getattr(args, "in_channels", 32)
        down_blocks = getattr(args, "downsampler_num_res_blocks", 2)
        up_blocks = getattr(args, "upsampler_num_res_blocks", 3)
        resnet_eps = getattr(args, "flux2_resnet_eps", 1e-6)
        resnet_act_fn = getattr(args, "flux2_resnet_act_fn", "swish")
        resnet_groups = getattr(args, "flux2_resnet_groups", 32)
        naive_down = getattr(args, "flux2_naive_downsample_mode", None)
        down_norm_act = getattr(args, "flux2_downsampler_add_norm_act_before_conv_out", True)
        up_norm_act = getattr(args, "flux2_upsampler_add_norm_act_before_conv_out", True)
        num_pre_upsampler = getattr(args, "flux2_upsampler_num_pre_upsampler_res_blocks", 0)
        down_add_attn = getattr(args, "flux2_downsampler_add_attention", False)
        down_attn_head_dim = getattr(args, "flux2_downsampler_attention_head_dim", None)
        down_attn_groups = getattr(args, "flux2_downsampler_attn_groups", None)
        up_add_attn = getattr(args, "flux2_upsampler_add_attention", False)
        up_attn_head_dim = getattr(args, "flux2_upsampler_attention_head_dim", None)
        up_attn_groups = getattr(args, "flux2_upsampler_attn_groups", None)
        naive_upsample = getattr(args, "flux2_naive_upsample_mode", None)
        token_upsampler = Flux2TokenUpsampler(
            in_channels=in_ch,
            num_res_blocks=up_blocks,
            num_pre_upsampler_res_blocks=num_pre_upsampler,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            add_norm_act_before_conv_out=up_norm_act,
            add_attention=up_add_attn,
            attention_head_dim=up_attn_head_dim,
            attn_groups=up_attn_groups,
            naive_upsample_mode=naive_upsample,
        )
        token_downsampler = Flux2TokenDownsampler(
            in_channels=in_ch,
            num_res_blocks=down_blocks,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            naive_downsample_mode=naive_down,
            add_norm_act_before_conv_out=down_norm_act,
            add_attention=down_add_attn,
            attention_head_dim=down_attn_head_dim,
            attn_groups=down_attn_groups,
        )
        print("<load_token_ae> ===> Loaded flux2 token AE")
    else:
        if TokenUpsampler is None or TokenDownsampler is None:
            raise ImportError("adapter_foveation_time_cond not found. Add image_generation to path or use --tokenAE_arch flux2.")
        in_ch = getattr(args, "in_channels", 32)
        feat_ch = getattr(args, "feature_channels", 64)
        emb_dim = getattr(args, "embedding_dim", 3072)
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
        print("<load_token_ae> ===> Loaded adapter token AE")

    if args.tokenAE_folder is not None:
        token_upsampler.load_state_dict(torch.load(os.path.join(args.tokenAE_folder, "token_upsampler.pt"), map_location="cpu"))
        token_downsampler.load_state_dict(torch.load(os.path.join(args.tokenAE_folder, "token_downsampler.pt"), map_location="cpu"))

    return token_upsampler.to(device).to(torch.bfloat16), token_downsampler.to(device).to(torch.bfloat16)

def flux2_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    
    # WandB logging arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use WandB for logging.")
    parser.add_argument("--wandb_project", type=str, default="flux2-training", help="WandB project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name.")
    parser.add_argument("--validation_prompts", type=str, nargs="+", default=None, help="List of validation prompts.")
    parser.add_argument("--validation_steps", type=int, default=500, help="Generate validation images every N steps.")
    parser.add_argument("--validation_height", type=int, default=1024, help="Height of validation images.")
    parser.add_argument("--validation_width", type=int, default=1024, help="Width of validation images.")
    parser.add_argument("--num_validation_images", type=int, default=1, help="Number of validation images per prompt.")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG guidance scale")
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # token AE arguments
    parser.add_argument("--tokenAE_folder", type=str, default=None, help="Path to token AE folder (loads tokenAE_config.yaml when present).")
    parser.add_argument("--tokenAE_arch", type=str, default="adapter", choices=["adapter", "flux2"],
                        help="TokenAE architecture. Overridden by tokenAE_config.yaml when loading from tokenAE_folder.")
    parser.add_argument("--in_channels", type=int, default=32, help="Token AE in_channels (latent channels).")
    parser.add_argument("--feature_channels", type=int, default=64, help="Token AE feature_channels (adapter only).")
    parser.add_argument("--embedding_dim", type=int, default=3072, help="Token AE embedding_dim (adapter only, FLUX2: 48*128).")
    parser.add_argument("--upsampler_num_res_blocks", type=int, default=10, help="Number of residual blocks for token upsampler.")
    parser.add_argument("--downsampler_num_res_blocks", type=int, default=1, help="Number of residual blocks for token downsampler (adapter). For flux2 default 2.")
    parser.add_argument("--tokenAE_dont_use_learned_components", action="store_true", help="Do not use learned components for token AE, only residual path (adapter only).")
    parser.add_argument("--tokenAE_learning_rate", type=float, default=1e-4, help="Learning rate for token AE.")
    parser.add_argument("--tokenAE_train_steps", type=int, default=None, help="Train only token AE for the first N steps, then switch to LoRA/DiT. None = train both throughout.")
    parser.add_argument("--upsampler_normalization_type", type=str, default="gn", choices=["ada", "gn", "none"], help="Normalization type for token upsampler (adapter only).")
    parser.add_argument("--downsampler_normalization_type", type=str, default="none", choices=["ada", "gn", "none"], help="Normalization type for token downsampler (adapter only).")
    parser.add_argument("--upsampler_variance_correction", type=float, default=None, help="Variance correction for token upsampler (adapter only).")
    parser.add_argument("--downsampler_variance_correction", type=float, default=None, help="Variance correction for token downsampler (adapter only).")
    parser.add_argument("--naive_upsample_mode", type=str, default="nearest", choices=["nearest", "bilinear"], help="Naive upsample mode for token upsampler (adapter only).")
    parser.add_argument("--naive_downsample_mode", type=str, default="nearest", choices=["nearest", "bilinear"], help="Naive downsample mode for token downsampler (adapter only).")
    parser.add_argument("--max_training_steps", type=int, default=None, help="Stop training after this many steps. None = train for full num_epochs.")

    # Flux2 TokenAE config (used when tokenAE_arch=flux2; overridden by tokenAE_config.yaml)
    parser.add_argument("--flux2_naive_downsample_mode", type=str, default=None,
                        help="Flux2 downsampler: None=learned, 'nearest'/'bilinear'=fixed interpolate.")
    parser.add_argument("--flux2_naive_upsample_mode", type=str, default=None, choices=["nearest", "bilinear"],
                        help="Flux2 upsampler: None=learned, 'nearest'/'bilinear'=fixed interpolate.")
    parser.add_argument("--flux2_resnet_eps", type=float, default=1e-6)
    parser.add_argument("--flux2_resnet_act_fn", type=str, default="swish")
    parser.add_argument("--flux2_resnet_groups", type=int, default=32)
    parser.add_argument("--flux2_downsampler_add_norm_act_before_conv_out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flux2_upsampler_add_norm_act_before_conv_out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flux2_upsampler_num_pre_upsampler_res_blocks", type=int, default=0)
    parser.add_argument("--flux2_downsampler_add_attention", action="store_true", default=False)
    parser.add_argument("--flux2_downsampler_attention_head_dim", type=int, default=None)
    parser.add_argument("--flux2_downsampler_attn_groups", type=int, default=None)
    parser.add_argument("--flux2_upsampler_add_attention", action="store_true", default=False)
    parser.add_argument("--flux2_upsampler_attention_head_dim", type=int, default=None)
    parser.add_argument("--flux2_upsampler_attn_groups", type=int, default=None)

    return parser


if __name__ == "__main__":
    parser = flux2_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
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
        )
    )
    # load token upsampler and downsampler
    token_upsampler, token_downsampler = load_token_ae(args, accelerator.device)
    # create training module
    token_ae_config = _token_ae_config_from_args(args)
    model = Flux2FoveatedImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        token_upsampler=token_upsampler,
        token_downsampler=token_downsampler,
        token_ae_config=token_ae_config,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=accelerator.device,
    )
    # Create logger (WandB or standard)
    if args.use_wandb:
        validation_prompts = args.validation_prompts if args.validation_prompts else []
        model_logger = WandbModelLogger(
            output_path=args.output_path,
            project_name=args.wandb_project,
            run_name=args.wandb_run_name,
            config=vars(args),
            remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
            validation_prompts=validation_prompts,
            validation_steps=args.validation_steps,
            num_validation_images=args.num_validation_images,
            validation_kwargs={
                "height": args.validation_height,
                "width": args.validation_width,
                "rand_device": "cuda",
                "num_inference_steps": args.num_inference_steps,
                "cfg_scale": args.cfg_scale,
                "seed": args.seed,
            },
        )
    else:
        model_logger = ModelLogger(
            args.output_path,
            remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "distill": launch_training_task, # added foveation distillation loss task
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
