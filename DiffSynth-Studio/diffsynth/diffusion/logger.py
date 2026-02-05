import os, torch
from accelerate import Accelerator
from typing import Optional, List, Callable, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


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
    
    def generate_validation_images(self, accelerator: Accelerator, model: torch.nn.Module):
        """Generate validation images using the model pipeline."""
        if not accelerator.is_main_process:
            return
        
        if not self.validation_prompts:
            return
        
        # Get the pipeline from the model
        unwrapped_model = accelerator.unwrap_model(model)
        if not hasattr(unwrapped_model, 'pipe'):
            return
        
        pipe = unwrapped_model.pipe
        
        # Switch to eval mode
        pipe.eval()
        
        all_images = []
        all_captions = []
        
        with torch.no_grad():
            for prompt in self.validation_prompts:
                try:
                    # Generate images
                    kwargs = {
                        "prompt": prompt,
                        **self.validation_kwargs
                    }
                    
                    # Call pipeline - adapt based on your pipeline's interface
                    images = pipe(**kwargs)
                    
                    if isinstance(images, (list, tuple)):
                        for i, img in enumerate(images[:self.num_validation_images]):
                            all_images.append(img)
                            all_captions.append(f"{prompt[:50]}... [{i}]" if len(prompt) > 50 else f"{prompt} [{i}]")
                    else:
                        all_images.append(images)
                        all_captions.append(prompt[:50] + "..." if len(prompt) > 50 else prompt)
                        
                except Exception as e:
                    print(f"Warning: Failed to generate validation image for prompt '{prompt[:30]}...': {e}")
        
        # Switch back to train mode
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
