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
    print("Warning: Could not import TokenUpsampler/TokenDownsampler from adapter_foveation_time_cond")
    print("Make sure the file is in the path or provide pre-trained weights via --tokenAE_folder")
    TokenUpsampler = None
    TokenDownsampler = None


def _token_ae_config_from_args(args):
    """Build token AE config dict for saving/loading (YAML-serializable, preserves None)."""
    return {
        "in_channels": getattr(args, "in_channels", 32),
        "feature_channels": getattr(args, "feature_channels", 64),
        "embedding_dim": getattr(args, "embedding_dim", 3072),
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
    """Load TokenUpsampler and TokenDownsampler."""
    if TokenUpsampler is not None and TokenDownsampler is not None:
        print("Creating new TokenAE (untrained)")
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
            variance_correction=args.upsampler_variance_correction
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
            variance_correction=args.downsampler_variance_correction
        )
    else:
        print("No TokenAE available, foveation will use nearest upsampling and downsampling")
        token_upsampler = None
        token_downsampler = None
    
    if args.tokenAE_folder is not None and token_upsampler is not None:
        token_upsampler.load_state_dict(torch.load(os.path.join(args.tokenAE_folder, "token_upsampler.pt"), map_location="cpu"))
        token_downsampler.load_state_dict(torch.load(os.path.join(args.tokenAE_folder, "token_downsampler.pt"), map_location="cpu"))
    
    token_upsampler = token_upsampler.to(device).to(torch.bfloat16)
    token_downsampler = token_downsampler.to(device).to(torch.bfloat16)
    
    return token_upsampler, token_downsampler

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
    parser.add_argument("--tokenAE_folder", type=str, default=None, help="Path to token AE folder.")
    parser.add_argument("--in_channels", type=int, default=32, help="Token AE in_channels (latent channels).")
    parser.add_argument("--feature_channels", type=int, default=64, help="Token AE feature_channels.")
    parser.add_argument("--embedding_dim", type=int, default=3072, help="Token AE embedding_dim (FLUX2: 48*128).")
    parser.add_argument("--upsampler_num_res_blocks", type=int, default=10, help="Number of residual blocks for token upsampler.")
    parser.add_argument("--downsampler_num_res_blocks", type=int, default=1, help="Number of residual blocks for token downsampler.")
    parser.add_argument("--tokenAE_dont_use_learned_components", action="store_true", help="Do not use learned components for token AE, only residual path.")
    parser.add_argument("--tokenAE_learning_rate", type=float, default=1e-4, help="Learning rate for token AE.")
    parser.add_argument("--tokenAE_train_steps", type=int, default=None, help="Train only token AE for the first N steps, then switch to LoRA/DiT. None = train both throughout.")
    parser.add_argument("--upsampler_normalization_type", type=str, default="gn", choices=["ada", "gn", "none"], help="Normalization type for token upsampler.")
    parser.add_argument("--downsampler_normalization_type", type=str, default="none", choices=["ada", "gn", "none"], help="Normalization type for token downsampler.")
    parser.add_argument("--upsampler_variance_correction", type=float, default=1.0, help="Variance correction for token upsampler.")
    parser.add_argument("--downsampler_variance_correction", type=float, default=1.0, help="Variance correction for token downsampler.")
    parser.add_argument("--naive_upsample_mode", type=str, default="nearest", choices=["nearest", "bilinear"], help="Naive upsample mode for token upsampler.")
    parser.add_argument("--naive_downsample_mode", type=str, default="nearest", choices=["nearest", "bilinear"], help="Naive downsample mode for token downsampler.")
    
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
