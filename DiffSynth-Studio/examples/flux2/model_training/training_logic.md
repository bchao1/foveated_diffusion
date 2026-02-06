# Foveated Flow Matching Training Logic

## Overview
- Foveated Flow Matching (FFM) is a training objective that combines Flow Matching (FM) with foveation.
- FFM is used to train the token upsampler and downsampler.
- FFM is defined in `diffsynth.diffusion.loss.FoveatedFlowMatchSFTLoss` in `diffsynth.diffusion.loss.py`.

## Training Objective
- **Main logic is self-contained in `diffsynth.diffusion.loss.py`**:
  - `FoveatedFlowMatchSFTLoss`: generates a random foveation mask each step, calls pipeline's foveated forward, computes MSE.
  - `_create_random_foveation_mask`: creates random circular masks for training (center and radius vary).
- The pipeline exposes `foveated_training_forward(inputs, timestep)` for the single-step foveated forward (minimal pipeline addition).
- For the foveation loss, the final noise prediction has the same high-resolution shape as the training target.
- For each training loop, the foveation mask is randomly generated and used for token upsampling and downsampling.