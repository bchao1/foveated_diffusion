Foveated image generation pipeline

# File organizations
- Inference API script: flux_foveation_tokenAE_inference.py
- Inference pipeline: pipeline_flux_foveation_x0pred.py
- Transformer model: transformer_flux_crpa.py

# High-level explanation and functional entrypoints
- Foveated generation
    - A foveation mask is given in the inference API script, which is then passed to the inference pipeline to determine which latents are kept as high-resolution and which are downsampled to become a lower resolution latents. 
    - The latents are upsampled and downsampled in the latent/unpacked domain, which is the latent space right after the VAE encoder, or right before the VAE decoder.
    - The latents are then packed into the packed domain, which involves rearranging the spatial and channel dimensions (pack_latents and unpack_latents) which is processed by the transformer_flux_crpa.py model.
- Token upsampling and downsampling
    - A token upsampler and downsampler in the latent/unpacked domain is used to upsample and downsample the latents.
    - The token downsampler operates on noisy latents.
    - The token upsampler operates on x0/clean image prediction, instead of velocity (pipeline_flux_foveation_x0pred.py, in upsample_x0_pred_unpacked).
    - The DiT outputs predicted velocity, but it is first converted to x0 prediction by upsample_x0_pred_unpacked, then converted back to velocity for rectified flow training and inference.
    - The token upsampler and downsampler design is defined in adapter_foveation_time_cond.py, which is transformer agnostic. Any network that does 2x down/upsampling can be used.
- Mixed-resolution denoising
    - The DiT processes downsampled + origianl high-res latents, hence the output predicted velocity is also in the mixed-resolution domain.
    - The upsampler upsamples the low-res parts, and merges with high-res parts to get the regular high-res grid.
    - For low-res parts to be processed, a downsample_remaining_high_res function is used to downsample the high-res parts to the same resolution as the low-res parts. This is defined in pipeline_flux_foveation_x0pred.py, in _downsample_remaining_high_res function. 
    - The high-res parts from DiT are directly used for the high-res parts of the final image.
    - The low-res parts from DiT are upsampled to the same resolution as the high-res parts, and merged with the high-res parts to get the final prediction.
- CRPA attention mechanism
    - The attention mechanism is modified using foveation mask to determine which key value pairs to attend to. This is defined in transformer_flux_crpa.py, in the FluxCRPAAttnProcessor class.
- VAE decoding
    - For the final VAE decoding step, it is done in the final block of __call__ in pipeline_flux_foveation_x0pred.py.
    - For direct, the final upsampled latents are directly decoded by the VAE.
    - For merge, the final upsampled latents are split into high-res and low-res parts, the low-res parts are upsampled to the same resolution as the high-res parts, using downsample_remaining_high_res function to get a regular low-res grid, and merged with the high-res parts to get the final prediction. 

# Goal
- The goal is to integrate this current pipeline into FLUX2, which has the following entrypoints:   
    - Inference pipeline: pipeline_flux2_klein.py
    - Transformer model: transformer_flux2.py

- In FLUX2 pipeline, there are some differences in the design
    - Latent image ids now is 4D instead of 2D. However, the 0th and 3rd dimensions are not used, so the 1st and 2nd are the spatial dimensions.
    - The packing and unpacking functions are named differently. In the current FLUX2 pipeline, patchify/unpatchify roughly does the same thing as pack/unpack in FLUX, which involves spatial and channel dimension rearrangement.
    - The packing and unpacking functions simply reshapes tenors from (B, C, H, W) to (B, H * W, C) and vice versa, to get a token list, instead of spatial grid.

# Final agent output
- Inference pipeline: pipeline_flux2_klein_foveation.py, which integrates the foveation logic into pipeline_flux2_klein.py
- Transformer model: transformer_flux2_foveation.py, which integrates the foveation logic into transformer_flux2.py