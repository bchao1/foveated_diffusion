# Token autoencoder pretraining

## Token upsampler and downsampler definition
- the token upsampler and downsampler are defined in the adapter_foveation_time_cond.py file
- this is called in train_foveated.py file, and the arguments are also passed to the train_foveated.py file

## goal of tokenAE pretraining
- the goal of tokenAE pretraining is to learn to compress and decompress the VAE latents
- the VAE latents is 2X downsampled by the downsampler, and then 2X upsampled by the upsampler
- the upsampled latents are then passed through the VAE to get the reconstructed image
- the reconstructed image is then compared to the original image to get the loss
- the loss is then backpropagated to the token upsampler and downsampler

## Image processing logic 
- the images should be handled in the same way as the images are handled in the train_foveated.py file

## Training logic
- VAE is loaded with how it is loaded in the train_foveated.py file and freezed. 
- only the token upsampler and downsampler are trained.
- trained with VAE-decoded image as the target


