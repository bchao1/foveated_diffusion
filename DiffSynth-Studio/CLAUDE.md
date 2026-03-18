# Foveated Diffusion Implementation
DiffSynth-Studio-based implementation of Foveated Diffusion built on FLUX2.

## Algorithm Overview

Foveated Diffusion is a perceptually motivated, mixed-resolution diffusion framework that reduces the token count during denoising by exploiting the human visual system's eccentricity-dependent acuity. Given a binary foveation mask `M`, high-resolution (HR) tokens are retained in the foveal region while peripheral regions are represented at lower resolution (1 LR token per 2×2 HR block), yielding a reduced token sequence of length `L = m + (h·w - m) / 4` where `m` is the number of HR tokens.

### Key Components

**Foveated Tokenization**
- HR latents: `z0_HR = E(x)` — VAE-encoded full-resolution image
- LR latents: `z0_LR = E(Down(x))` — VAE-encoded bicubically downsampled image
- Clean foveated target: `z0_fov = Merge(z0_HR, z0_LR, M)` — mixed-resolution token sequence consistent by construction

**Foveated Generation (Inference)**
- Sample Gaussian noise in the foveated token space at reduced length `L`
- Iteratively denoise using a flow-matching DiT operating on the mixed-resolution token sequence
- Split clean tokens: `(z0_HR, z0_LR) = Split(z0_fov, M)`
- Decode each separately with the VAE decoder and blend: `x_fov = M' ⊙ D(z0_HR) + (1 − M') ⊙ Up(D(z0_LR))`

**Mixed-Resolution RoPE (CRPA)**
- Standard RoPE assumes a uniform grid; foveated tokens break this assumption
- Following Wu et al. (CRPA), RoPE phases are aligned by resolution: LR key tokens are sub-sampled from full-resolution positions for HR queries, and HR key indices are normalized to the LR grid for LR queries

**Foveated Training**
- Post-training via LoRA on a pretrained FLUX2 DiT
- Flow-matching objective applied entirely on the variable-length foveated token sequence
- Masking strategies: randomized masks (gaze-agnostic), saliency-guided masks, bounding-box masks

### Implementation
- `Flux2FoveatedImagePipeline` (`flux2_image_foveated.py`) loads a `Flux2DiTFoveated` model (or converts a base FLUX2 DiT via CRPA attention processors) and orchestrates the foveated denoising loop
- Saliency masks are computed using `deepgaze_pytorch`
- Gaussian-blurred mask boundaries are supported for smooth fovea/periphery transitions

## Environments
- Use the `imp_tome` conda environment.

## References
- [context](./context.tex) file describes the Foveated Diffusion Algorithm in high-level.
- [Foveated Diffusion Pipeline](./diffsynth/pipelines/flux2_image_foveated.py) defines the foveated diffusion pipeline built on FLUX2.
- [Normal Diffusion Pipeline](./diffsynth/pipelines/flux2_image.py) defines the normal FLUX2 image generation pipeline.