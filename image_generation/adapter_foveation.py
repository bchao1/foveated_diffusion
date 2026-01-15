import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class TokenDownsampler(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim=None, num_res_blocks=2):
        super().__init__()
        
        # 1. Learned Branch (CNN)
        layers = []
        for _ in range(num_res_blocks):
            # Note: Residual blocks usually keep dim constant. 
            # If in_channels != out_channels, we might need a projection first, 
            # but for this specific "refinement" use-case, usually in==out.
            layers.append(ResidualBlock(in_channels))
            
        self.res_blocks = nn.Sequential(*layers)
        
        # The strided convolution that actually reduces dimension
        self.downsample = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )

        # 2. Projection for Naive Branch (if channels change)
        self.use_projection = in_channels != out_channels
        if self.use_projection:
            self.naive_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 3. Initialization
        self.apply(self._init_weights)
        # ZERO-INIT the final layer of the learned branch so it starts as 0 contribution
        nn.init.zeros_(self.downsample.weight)
        if self.downsample.bias is not None:
            nn.init.zeros_(self.downsample.bias)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, temb=None):
        # Path A: Naive (Average Pooling)
        # We assume "variance preserving" average pooling often used in diffusion
        # (Avg * 2) if fitting a specific distribution, but standard Avg is safer as baseline.
        # Check if your specific VAE/DiT expects variance correction. Standard here:
        x_naive = F.avg_pool2d(x, kernel_size=2, stride=2) * 2
        
        if self.use_projection:
            x_naive = self.naive_proj(x_naive)

        # Path B: Learned Residual
        x_learned = self.res_blocks(x)
        x_learned = self.downsample(x_learned)

        # Output = Naive + Residual
        return x_naive + x_learned

class TokenUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim=None, num_res_blocks=2, naive_upsample_mode="bilinear"):
        super().__init__()

        self.naive_upsample_mode = naive_upsample_mode

        # 1. Learned Branch (CNN)
        self.res_blocks = []
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(out_channels))
        self.res_blocks = nn.ModuleList(self.res_blocks)
        
        # Transposed conv to increase resolution
        self.upsample = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )

        # 2. Projection for Naive Branch (if channels change)
        self.use_projection = in_channels != out_channels
        if self.use_projection:
            self.naive_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 3. Initialization
        self.apply(self._init_weights)
        # ZERO-INIT the final layer so it starts as 0 contribution
        nn.init.zeros_(self.upsample.weight)
        if self.upsample.bias is not None:
            nn.init.zeros_(self.upsample.bias)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, temb=None):
        # Path A: Naive (Bilinear Upsampling)
        if self.naive_upsample_mode == "bilinear":
            x_naive = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False) * 0.5
        elif self.naive_upsample_mode == "nearest":
            x_naive = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            raise ValueError(f"Naive Upsample mode {self.naive_upsample_mode} is not supported!")

        if self.use_projection:
            x_naive = self.naive_proj(x_naive)

        # Path B: Learned Residual
        x_learned = self.upsample(x)
        for block in self.res_blocks:
            x_learned = block(x_learned)

        # Output = Naive + Residual
        return x_naive #+ x_learned

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
    downsampler = TokenDownsampler(in_channels=128, out_channels=128, num_res_blocks=10).to(device)
    upsampler = TokenUpsampler(in_channels=128, out_channels=128, num_res_blocks=10).to(device)

    # Forward Pass
    x_down = downsampler(x)
    x_up = upsampler(x_down)

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
    x_naive_down = F.avg_pool2d(x, kernel_size=2, stride=2) * 2
    diff = (x_down - x_naive_down).abs().sum()
    print(f"Difference from naive avg pool after init: {diff.item():.6f} (Should be close to 0)")
    
    print("Test passed")

    # print number of parameters in M
    count_parameters(downsampler)
    count_parameters(upsampler)