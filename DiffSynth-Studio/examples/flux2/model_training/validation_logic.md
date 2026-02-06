# Foveated Flow Matching Validation Logic

## Overview
- The logging functions are fully defined in `diffsynth.diffusion.logger.py`
- The validation logic is implemented in `WandbModelLogger.generate_validation_images`

## Implementation
- **Foveated generation**: When the pipeline has `token_upsampler` and `token_downsampler`, the logger generates 4 images per prompt with foveation masks at left, right, top, bottom (center positions: (-0.3,0), (0.3,0), (0,-0.3), (0,0.3)).
- **Foveation mask visualization**: The four masks are logged once per validation run as `validation/foveation_masks`.
- **Target and noise prediction visualization**: One flow-matching training step is run; target and noise_pred are unpatchified from [B, H*W, C] to spatial format, first 3 channels visualized with the same scale (min/max from target) and colormap via `_tensor_to_pil_vis` and `_unpatchify_for_vis`.