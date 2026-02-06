import os, torch
import numpy as np
from PIL import Image
from accelerate import Accelerator
from typing import Optional, List, Callable, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _create_foveation_mask(h: int, w: int, center: tuple, r: float, device):
    """Create circular foveation mask. center in (-0.5, 0.5), r relative to half-diagonal."""
    cx = (center[0] + 0.5) * w
    cy = (center[1] + 0.5) * h
    diagonal = (h**2 + w**2) ** 0.5
    radius_px = r * (diagonal / 2.0)
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2
    return mask


def _unpatchify_for_vis(latents: torch.Tensor, latent_height: int, latent_width: int):
    """Convert [B, H*W, C] packed FLUX2 latents to [B, 3, H', W'] for visualization (first 3 channels)."""
    B, _, C = latents.shape
    latents = latents.reshape(B, latent_height, latent_width, C).permute(0, 3, 1, 2)
    if C % 4 != 0:
        return latents[:, : min(3, C)].float().cpu()
    c_base = C // 4
    latents = latents.reshape(B, c_base, 2, 2, latent_height, latent_width)
    latents = latents.permute(0, 1, 4, 2, 5, 3).reshape(B, c_base, latent_height * 2, latent_width * 2)
    return latents[:, : min(3, c_base)].float().cpu()


def _tensor_to_pil_vis(x: torch.Tensor, vmin: float, vmax: float):
    """Convert tensor [B, 3, H, W] to PIL with fixed scale. No normalization."""
    x = x.clamp(vmin, vmax)
    x = (x - vmin) / (vmax - vmin + 1e-8)
    x = (x * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(x[i]) for i in range(x.shape[0])]


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_training_start(self, accelerator: Accelerator, model: torch.nn.Module):
        """Called at the very beginning of training (step 0), before any finetuning."""
        pass


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            state_dict = accelerator.get_state_dict(model)
            state_dict = unwrapped.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
            if hasattr(unwrapped, "save_token_ae") and callable(getattr(unwrapped, "save_token_ae")):
                token_ae_path = os.path.join(self.output_path, "tokenAE")
                unwrapped.save_token_ae(token_ae_path, step=self.num_steps)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            state_dict = accelerator.get_state_dict(model)
            state_dict = unwrapped.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            # Save token upsampler/downsampler to tokenAE subdirectory if applicable
            if hasattr(unwrapped, "save_token_ae") and callable(getattr(unwrapped, "save_token_ae")):
                token_ae_path = os.path.join(self.output_path, "tokenAE")
                unwrapped.save_token_ae(token_ae_path, step=self.num_steps)


class WandbModelLogger(ModelLogger):
    """
    Extended ModelLogger with WandB integration for logging training losses 
    and generating validation images using the model pipeline.
    """
    def __init__(
        self, 
        output_path: str,
        project_name: str = "diffsynth-training",
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        remove_prefix_in_ckpt: Optional[str] = None,
        state_dict_converter: Callable = lambda x: x,
        validation_prompts: Optional[List[str]] = None,
        validation_steps: int = 500,
        log_image_steps: int = 500,
        num_validation_images: int = 4,
        validation_kwargs: Optional[dict] = None,
    ):
        """
        Args:
            output_path: Path to save model checkpoints.
            project_name: WandB project name.
            run_name: WandB run name (optional).
            config: Configuration dict to log to WandB.
            remove_prefix_in_ckpt: Prefix to remove from checkpoint keys.
            state_dict_converter: Function to convert state dict before saving.
            validation_prompts: List of prompts for validation image generation.
            validation_steps: Generate validation images every N steps.
            log_image_steps: Log images every N steps.
            num_validation_images: Number of validation images to generate per prompt.
            validation_kwargs: Additional kwargs for pipeline inference.
        """
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
        
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Please install it with: pip install wandb")
        
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.validation_prompts = validation_prompts or []
        self.validation_steps = validation_steps
        self.log_image_steps = log_image_steps
        self.num_validation_images = num_validation_images
        self.validation_kwargs = validation_kwargs or {}
        
        self._wandb_initialized = False
        self._loss_buffer = []
    
    def init_wandb(self, accelerator: Accelerator):
        """Initialize WandB run (only on main process)."""
        if self._wandb_initialized:
            return
            
        if accelerator.is_main_process:
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=self.config,
                resume="allow",
            )
            self._wandb_initialized = True
    
    def log_loss(self, loss: float, step: Optional[int] = None):
        """Log training loss to WandB."""
        if not self._wandb_initialized:
            return
        
        step = step if step is not None else self.num_steps
        wandb.log({"train/loss": loss}, step=step)
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log arbitrary metrics to WandB."""
        if not self._wandb_initialized:
            return
        
        step = step if step is not None else self.num_steps
        wandb.log(metrics, step=step)
    
    def log_images(self, images: List[Any], captions: Optional[List[str]] = None, key: str = "validation"):
        """Log images to WandB."""
        if not self._wandb_initialized:
            return
        
        if captions is None:
            captions = [f"Image {i}" for i in range(len(images))]
        
        wandb_images = [wandb.Image(img, caption=cap) for img, cap in zip(images, captions)]
        wandb.log({key: wandb_images}, step=self.num_steps)
    
    def _is_foveated_pipeline(self, pipe):
        """Check if pipeline supports foveated generation."""
        return (
            hasattr(pipe, "token_upsampler")
            and hasattr(pipe, "token_downsampler")
            and pipe.token_upsampler is not None
            and pipe.token_downsampler is not None
        )

    def _log_foveation_masks_once(self, pipe, height, width):
        """Visualize four foveation masks (left, right, top, bottom) once."""
        h, w = height // 16, width // 16
        device = getattr(pipe, "device", "cuda")
        if hasattr(device, "index"):
            device = device
        positions = [("left", (-0.3, 0)), ("right", (0.3, 0)), ("top", (0, -0.3)), ("bottom", (0, 0.3))]
        mask_images = []
        for name, center in positions:
            mask = _create_foveation_mask(h, w, center, r=0.25, device=device)
            mask_np = mask.float().cpu().numpy() * 255
            mask_pil = Image.fromarray(mask_np.astype(np.uint8))
            mask_images.append(wandb.Image(mask_pil, caption=f"foveation_mask_{name}"))
        if mask_images:
            wandb.log({"validation/foveation_masks": mask_images}, step=self.num_steps)

    def _visualize_target_noise_pred(self, pipe, height, width):
        """Run flow-matching steps at timesteps 0, middle, last and visualize target vs noise_pred with shared scale."""
        latent_height, latent_width = height // 16, width // 16
        device = getattr(pipe, "device", "cuda")
        dtype = getattr(pipe, "torch_dtype", torch.bfloat16)
        if not hasattr(pipe, "foveated_training_forward"):
            return
        try:
            prompt = self.validation_prompts[0] if self.validation_prompts else ""
            seed = self.validation_kwargs.get("seed", 42)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            pipe.scheduler.set_timesteps(1000, training=True)
            ts = pipe.scheduler.timesteps
            # 25%, 50%, 75% timesteps for tracking validation
            timestep_ids = [int(len(ts) * 0.25), int(len(ts) * 0.50), int(len(ts) * 0.75)]
            labels = ["t0", "t_mid", "t_last"]
            all_target_vis = []
            all_pred_vis = []
            all_l2_error = []
            inputs_shared_base = {
                "height": height, "width": width, "prompt": prompt, "cfg_scale": 1.0,
                "embedded_guidance": 1.0, "input_image": None, "rand_device": str(device),
                "seed": seed, **{k: v for k, v in self.validation_kwargs.items() if k not in ("height", "width")},
            }
            inputs_posi, inputs_nega = {"prompt": prompt}, {"negative_prompt": ""}
            for unit in pipe.units:
                inputs_shared_base, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared_base, inputs_posi, inputs_nega)

            input_latents = pipe.fixed_clean_latent # for tracking validation
            noise = torch.randn_like(input_latents)
            foveation_mask = _create_foveation_mask(latent_height, latent_width, center=(0.0, 0.0), r=0.25, device=device)

            for tid, label in zip(timestep_ids, labels):
                timestep_id = torch.tensor([tid], device=ts.device)
                timestep = ts[timestep_id].to(dtype=dtype, device=device)
                latents = pipe.scheduler.add_noise(input_latents, noise, timestep)
                training_target = pipe.scheduler.training_target(input_latents, noise, timestep)
                inputs_shared = dict(inputs_shared_base)
                inputs_shared["latents"] = latents
                inputs_shared["input_latents"] = input_latents
                inputs_shared["foveation_mask"] = foveation_mask
                inputs = dict(inputs_shared, **inputs_posi)
                noise_pred = pipe.foveated_training_forward(inputs, timestep, timestep_id)
                target_vis = _unpatchify_for_vis(training_target, latent_height, latent_width)
                pred_vis = _unpatchify_for_vis(noise_pred, latent_height, latent_width)
                l2_error = (target_vis - pred_vis).pow(2).sum(dim=1, keepdim=True).sqrt()
                all_target_vis.append((label, target_vis))
                all_pred_vis.append((label, pred_vis))
                all_l2_error.append((label, l2_error))
            
            # visualize target and noise prediction
            target_images = []
            pred_images = []
            for (label, t), (_, p) in zip(all_target_vis, all_pred_vis):
                vmin = float(min(t.min().item(), p.min().item()))
                vmax = float(max(t.max().item(), p.max().item()))
                target_images.append(wandb.Image(_tensor_to_pil_vis(t, vmin, vmax)[0], caption=f"target_{label}"))
                pred_images.append(wandb.Image(_tensor_to_pil_vis(p, vmin, vmax)[0], caption=f"noise_pred_{label}"))
            # visualize l2 error
            l2_images = []
            for label, e in all_l2_error:
                e_3ch = e.expand(-1, 3, -1, -1)
                l2_max = float(e_3ch.max().item())
                l2_images.append(wandb.Image(_tensor_to_pil_vis(e_3ch, 0.0, l2_max)[0], caption=f"l2_error_{label}"))
            wandb.log({
                "validation/target": target_images,
                "validation/noise_pred": pred_images,
                "validation/l2_error": l2_images,
            }, step=self.num_steps)
        except Exception as e:
            print(f"Warning: Target/noise viz failed: {e}")

    def generate_validation_images(self, accelerator: Accelerator, model: torch.nn.Module):
        """Generate validation images. Supports foveated pipeline with 4-position masks and target/noise viz."""
        if not accelerator.is_main_process:
            return
        if not self.validation_prompts:
            return
        unwrapped_model = accelerator.unwrap_model(model)
        if not hasattr(unwrapped_model, "pipe"):
            return
        pipe = unwrapped_model.pipe
        pipe.eval()
        all_images = []
        all_captions = []
        use_foveation = self._is_foveated_pipeline(pipe)
        height = self.validation_kwargs.get("height", 1024)
        width = self.validation_kwargs.get("width", 1024)

        with torch.no_grad():
            if use_foveation and hasattr(pipe, "foveated_training_forward"):
                self._visualize_target_noise_pred(pipe, height, width)

            if use_foveation:
                self._log_foveation_masks_once(pipe, height, width)
                positions = [("left", (-0.3, 0)), ("right", (0.3, 0)), ("top", (0, -0.3)), ("bottom", (0, 0.3))]
                h, w = height // 16, width // 16
                device = getattr(pipe, "device", "cuda")
                for prompt in self.validation_prompts:
                    for pos_name, center in positions:
                        try:
                            mask = _create_foveation_mask(h, w, center, r=0.25, device=device)
                            kwargs = {"prompt": prompt, "foveation_mask": mask, "token_upsampler": pipe.token_upsampler, "token_downsampler": pipe.token_downsampler, **self.validation_kwargs}
                            images = pipe(**kwargs)
                            img = images[0] if isinstance(images, (list, tuple)) else images
                            all_images.append(img)
                            all_captions.append(f"{prompt[:40]}... [{pos_name}]" if len(prompt) > 40 else f"{prompt} [{pos_name}]")
                        except Exception as e:
                            print(f"Warning: Foveated validation failed {pos_name}: {e}")
            else:
                for prompt in self.validation_prompts:
                    try:
                        kwargs = {"prompt": prompt, **self.validation_kwargs}
                        images = pipe(**kwargs)
                        if isinstance(images, (list, tuple)):
                            for i, img in enumerate(images[: self.num_validation_images]):
                                all_images.append(img)
                                all_captions.append(f"{prompt[:50]}... [{i}]" if len(prompt) > 50 else f"{prompt} [{i}]")
                        else:
                            all_images.append(images)
                            all_captions.append(prompt[:50] + "..." if len(prompt) > 50 else prompt)
                    except Exception as e:
                        print(f"Warning: Validation failed: {e}")


        pipe.train()
        if all_images:
            self.log_images(all_images, all_captions, key="validation/generated")
    
    def on_training_start(self, accelerator: Accelerator, model: torch.nn.Module):
        """Called at the very beginning of training (step 0), before any finetuning."""
        # Initialize WandB
        self.init_wandb(accelerator)
        
        # Generate baseline validation images at step 0
        if accelerator.is_main_process:
            print("Generating baseline validation images (step 0)...")
        self.generate_validation_images(accelerator, model)
    
    def on_step_end(
        self, 
        accelerator: Accelerator, 
        model: torch.nn.Module, 
        save_steps: Optional[int] = None,
        loss: Optional[float] = None,
        **kwargs
    ):
        """Called at the end of each training step."""
        # Initialize WandB if not done yet
        self.init_wandb(accelerator)
        
        self.num_steps += 1
        
        # Log loss
        if loss is not None and accelerator.is_main_process:
            self.log_loss(loss)
        
        # Log additional metrics
        if kwargs and accelerator.is_main_process:
            metrics = {f"train/{k}": v for k, v in kwargs.items() if isinstance(v, (int, float))}
            if metrics:
                self.log_metrics(metrics)
        
        # Save model checkpoint
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        
        # Generate validation images
        if self.validation_steps > 0 and self.num_steps % self.validation_steps == 0:
            self.generate_validation_images(accelerator, model)
    
    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id: int):
        """Called at the end of each epoch."""
        # Initialize WandB if not done yet
        self.init_wandb(accelerator)
        
        # Log epoch
        if accelerator.is_main_process:
            wandb.log({"epoch": epoch_id}, step=self.num_steps)
        
        # Save model
        super().on_epoch_end(accelerator, model, epoch_id)
        
        # Generate validation images at end of epoch
        self.generate_validation_images(accelerator, model)
    
    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps: Optional[int] = None):
        """Called when training ends."""
        super().on_training_end(accelerator, model, save_steps)
        
        # Final validation images
        self.generate_validation_images(accelerator, model)
        
        # Finish WandB run
        if accelerator.is_main_process and self._wandb_initialized:
            wandb.finish()
