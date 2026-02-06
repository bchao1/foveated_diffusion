# DiffSynth training loop structure


## Overall structure
- Training entrypoint: `train_foveated.py`
- Training module: `Flux2FoveatedImageTrainingModule`, which inherits from `DiffusionTrainingModule` in `diffsynth.diffusion.training_module`
- Training model: `Flux2FoveatedImagePipeline` in `diffsynth.pipelines.flux2_image_foveated`
- Training job is launched by `launch_training_task` in `diffsynth.diffusion.runner`

## Goal
- TokenUpsampler and TokenDownsampler are trained.
- Need to extend the current training pipeline to support the following:
    - saving token upsampler and downsampler to a separate folder during training (`tokenAE_folder` subdirectory in the output path), hence they cannot be saved as part of the model checkpoint (`.safetensors`)
    - setting the learning rate for token upsampler and downsampler separately from the main model, which is defined in `launch_training_task`

