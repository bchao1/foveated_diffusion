import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaGroupNorm(nn.Module):
    """
    The CNN equivalent of Flux's AdaLayerNorm.
    Uses GroupNorm instead of LayerNorm, but driven by the same 'temb'.
    """
    def __init__(self, embedding_dim, out_channels, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        
        # 1. Standard GroupNorm (affine=False because we predict the affine parameters)
        self.norm = nn.GroupNorm(num_groups, out_channels, affine=False, eps=eps)
        
        # 2. Projection Layer (The "Adapter" part)
        # Projects the large Flux embedding (e.g., 3072) down to 2 * CNN_channels (Scale + Shift)
        self.proj = nn.Linear(embedding_dim, out_channels * 2)
        
        # 3. Activation (Silu) for the embedding before projection
        self.act = nn.SiLU()

        # 4. Zero-init the projection so training starts as a standard GroupNorm
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, temb):
        # x: [Batch, Channels, H, W]
        # temb: [Batch, embedding_dim] (From Flux Transformer)

        # Apply GroupNorm
        x_norm = self.norm(x)

        # Project embedding to Scale and Shift
        # We apply SiLU to temb first, matching standard DiT/Flux practices
        style = self.proj(self.act(temb)) 
        
        # Reshape for broadcasting over spatial dims: [Batch, 2*C, 1, 1]
        style = style[:, :, None, None]
        scale, shift = style.chunk(2, dim=1)

        # Modulate
        # (1 + scale) ensures we start at identity when scale is 0
        x = x_norm * (1 + scale) + shift
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, embedding_dim, num_groups=1, use_ada_norm=True):
        """
        Args:
            channels: Number of input/output channels.
            embedding_dim: Dimension of the time embedding (temb).
            num_groups: Number of groups for GroupNorm.
            use_ada_norm: If True, uses AdaGroupNorm (time-conditioned).
                          If False, uses standard GroupNorm (ignores time).
        """
        super().__init__()
        self.use_ada_norm = use_ada_norm
        
        # 1. Select Normalization Type
        if self.use_ada_norm:
            # Your original time-conditioned norm
            print("ada norm")
            self.gn1 = AdaGroupNorm(embedding_dim, channels, num_groups)
            self.gn2 = AdaGroupNorm(embedding_dim, channels, num_groups)
        else:
            # Standard GroupNorm (sanity check mode)
            # affine=True ensures it can learn scale/shift immediately
            print("standard group norm")
            self.gn1 = nn.GroupNorm(num_groups, channels, affine=True)
            self.gn2 = nn.GroupNorm(num_groups, channels, affine=True)

        # 2. Standard Convolutional Layers (Unchanged)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act2 = nn.SiLU()

    def forward(self, x, temb):
        residual = x
        
        # Block 1
        if self.use_ada_norm:
            x = self.gn1(x, temb)
        else:
            x = self.gn1(x) # Ignores temb
            
        x = self.act1(x)
        x = self.conv1(x)
        
        # Block 2
        if self.use_ada_norm:
            x = self.gn2(x, temb)
        else:
            x = self.gn2(x) # Ignores temb
            
        x = self.act2(x)
        x = self.conv2(x)
        
        return residual + x

class TokenDownsampler(nn.Module):
    def __init__(self, in_channels, feature_channels, embedding_dim, num_res_blocks=2, learn_residual=False, variance_correction=False, use_ada_norm=True):
        super().__init__()

        self.variance_correction = variance_correction
        self.learn_residual = learn_residual

        self.input_proj = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)
        
        # 1. Learned Branch (CNN)
        self.res_blocks = []
        for _ in range(num_res_blocks):
            # Note: Residual blocks usually keep dim constant. 
            # If in_channels != out_channels, we might need a projection first, 
            # but for this specific "refinement" use-case, usually in==out.
            self.res_blocks.append(ResidualBlock(feature_channels, embedding_dim, use_ada_norm=use_ada_norm))
        self.res_blocks = nn.ModuleList(self.res_blocks)
            
        # The strided convolution that actually reduces dimension
        """
        self.downsample = nn.Conv2d(
            feature_channels,
            in_channels,
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        """

        self.downsample = nn.Sequential(
            # 1. Anti-aliased Downsampling (AvgPool preserves smooth gradients)
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # 2. Convolution (Stride 1 - pure processing, no shrinking)
            nn.Conv2d(feature_channels, in_channels, kernel_size=3, padding=1)
        )

        # 3. Initialization
        self.apply(self._init_weights)
        
        # ZERO-INIT the final layer of the learned branch so it starts as 0 contribution
        #nn.init.zeros_(self.downsample.weight)
        #if self.downsample.bias is not None:
        #    nn.init.zeros_(self.downsample.bias)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, temb):
        # Path A: Naive (Average Pooling)
        # We assume "variance preserving" average pooling often used in diffusion
        # (Avg * 2) if fitting a specific distribution, but standard Avg is safer as baseline.
        # Check if your specific VAE/DiT expects variance correction. Standard here:
        if self.learn_residual:
            x_naive = F.avg_pool2d(x, kernel_size=2, stride=2)
            if self.variance_correction:
                x_naive *= 2 # used in first-step of the training-free results
        else:
            x_naive = 0

        # Path B: Learned Residual
        x = self.input_proj(x) # expand channels
        for block in self.res_blocks:
            x = block(x, temb)
        x = self.downsample(x)

        # Output = Naive + Residual
        return x_naive + x

class UpsampleBilinear(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        
class TokenUpsampler(nn.Module):
    def __init__(self, 
        in_channels, 
        feature_channels,
        embedding_dim, 
        num_res_blocks=2, 
        cnn_upsample_mode="bilinear", 
        naive_upsample_mode="bilinear",
        learn_residual=False,
        variance_correction=False,
        use_ada_norm=True
    ):
        super().__init__()

        self.learn_residual = learn_residual
        self.cnn_upsample_mode = cnn_upsample_mode
        self.naive_upsample_mode = naive_upsample_mode
        
        # 1. Learned Branch (CNN)
        self.res_blocks = []
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(feature_channels, embedding_dim, use_ada_norm=use_ada_norm))
        self.res_blocks = nn.ModuleList(self.res_blocks)

        self.input_proj = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)
        self.output_proj = nn.Conv2d(feature_channels, in_channels, kernel_size=3, padding=1)
        
        # Transposed conv to increase resolution
        if self.cnn_upsample_mode == "conv":
            self.upsample = nn.ConvTranspose2d(
                feature_channels, 
                feature_channels,
                kernel_size=4, 
                stride=2, 
                padding=1
            )
        elif self.cnn_upsample_mode == "bilinear":
            self.upsample = UpsampleBilinear(scale_factor=2)
        else:
            raise ValueError(f"CNN Upsample mode {cnn_upsample_mode} is not supported!")

        # 3. Initialization
        self.apply(self._init_weights)


        # TODO: with new model, do not zero-init weights
        # ZERO-INIT the final layer so it starts as 0 contribution
        #nn.init.zeros_(self.upsample.weight)
        #if self.upsample.bias is not None:
        #     nn.init.zeros_(self.upsample.bias)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, temb):
        
        # Path A: Naive (Bilinear Upsampling or straight-through)
        if self.learn_residual:
            if self.naive_upsample_mode == "bilinear":
                x_naive = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False) * 0.5
            elif self.naive_upsample_mode == "nearest":
                x_naive = F.interpolate(x, scale_factor=2, mode="nearest")
            else:
                raise ValueError(f"Naive Upsample mode {self.naive_upsample_mode} is not supported!")
        else:
            x_naive = 0

        # Path B: Learned Residual
        x = self.input_proj(x)
        x = self.upsample(x)
        for block in self.res_blocks:
            x = block(x, temb)
        x = self.output_proj(x)

        # Output = Naive + Residual
        return x_naive + x

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # print in M
    print("Number of parameters in {}: {:.2f}M".format(model.__class__.__name__, total_params / 1e6))
    print("Number of trainable parameters in {}: {:.2f}M".format(model.__class__.__name__, trainable_params / 1e6))

if __name__ == "__main__":
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 128, 64, 64).to(device)

    # Instantiate
    downsampler = TokenDownsampler(in_channels=128, out_channels=128, embedding_dim=3072, num_res_blocks=2).to(device)
    upsampler = TokenUpsampler(in_channels=128, out_channels=128, embedding_dim=3072, num_res_blocks=2, naive_upsample_mode="nearest").to(device)
    temb = torch.randn(1, 3072).to(device)

    # Forward Pass
    x_down = downsampler(x, temb)
    x_up = upsampler(x_down, temb)

    # Verifications
    print(f"Original shape: {x.shape}")
    print(f"Downsampled shape: {x_down.shape}")
    print(f"Upsampled shape: {x_up.shape}")

    assert x_down.shape[-2] * 2 == x.shape[-2], "Downsampled latent shape incorrect"
    assert x_down.shape[-1] * 2 == x.shape[-1], "Downsampled latent shape incorrect"
    assert x_up.shape == x.shape, "Reconstructed latent shape incorrect"
    
    # Verify Residual Initialization (Expect roughly naive output initially)
    # Since we zero-init the learned branch, x_down should be exactly AvgPool(x) initially
    # (ignoring float precision slightly)
    x_naive_down = F.avg_pool2d(x, 2)
    diff = (x_down - x_naive_down).abs().sum()
    print(f"Difference from naive avg pool after init: {diff.item():.6f} (Should be close to 0)")
    
    print("Test passed")

    # print number of parameters in M
    count_parameters(downsampler)
    count_parameters(upsampler)