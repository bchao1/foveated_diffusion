# DiffSynth training loop structure


## Overall structure
- Training entrypoint: `train_foveated.py`
- Training module: `Flux2FoveatedImageTrainingModule`, which inherits from `DiffusionTrainingModule` in `diffsynth.diffusion.training_module`
- Training model: `Flux2FoveatedImagePipeline` in `diffsynth.pipelines.flux2_image_foveated`
- Training job is launched by `launch_training_task` in `diffsynth.diffusion.runner`

## Token upsampler and downsampler integration

TokenUpsampler and TokenDownsampler are trained with the following behavior:

### 1. Trainability
- When `token_upsampler` and `token_downsampler` are provided to `Flux2FoveatedImageTrainingModule`, they are automatically added to the trainable models list (in addition to any `--trainable_models` or LoRA targets).

### 2. Separate checkpoint saving
- Token upsampler and downsampler are **excluded** from the main `.safetensors` checkpoint.
- They are saved to a separate `tokenAE` subdirectory under the output path, with step-specific folders:
  - `{output_path}/tokenAE/step-{N}/token_upsampler.pt`
  - `{output_path}/tokenAE/step-{N}/token_downsampler.pt`
- This happens on each checkpoint save (both step-based and epoch-based) via `ModelLogger.save_model` and `ModelLogger.on_epoch_end`.

### 3. Separate learning rate
- Use `--tokenAE_learning_rate` (default: 1e-4) to set the learning rate for token upsampler and downsampler.
- The main model uses `--learning_rate` (default: 1e-4).
- `launch_training_task` uses parameter groups to apply these learning rates separately.

