import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple

# Optional: DiT block components from DiffSynth (required for TokenDiTUpsampler / TokenDiTDownsampler)
try:
    from diffsynth.models.flux2_dit import (
        Flux2SingleTransformerBlock,
        Flux2Modulation,
        Flux2PosEmbed,
    )
    _HAS_FLUX2_DIT = True
except ImportError:
    Flux2SingleTransformerBlock = None
    Flux2Modulation = None
    Flux2PosEmbed = None
    _HAS_FLUX2_DIT = False

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
    def __init__(self, channels, embedding_dim, num_groups=1, normalization_type: Literal["ada", "gn", "none"] = "gn"):
        """
        Args:
            channels: Number of input/output channels.
            embedding_dim: Dimension of the time embedding (temb).
            num_groups: Number of groups for GroupNorm.
            normalization_type: If "ada", uses AdaGroupNorm (time-conditioned).
                          If False, uses standard GroupNorm (ignores time).
        """
        super().__init__()
        self.normalization_type = normalization_type
        
        # 1. Select Normalization Type
        if normalization_type == "ada":
            # Your original time-conditioned norm
            print("ada norm")
            self.gn1 = AdaGroupNorm(embedding_dim, channels, num_groups)
            self.gn2 = AdaGroupNorm(embedding_dim, channels, num_groups)
        elif normalization_type == "gn":
            # Standard GroupNorm (sanity check mode)
            # affine=True ensures it can learn scale/shift immediately
            print("standard group norm")
            self.gn1 = nn.GroupNorm(num_groups, channels, affine=True)
            self.gn2 = nn.GroupNorm(num_groups, channels, affine=True)
        elif normalization_type == "none":
            print("no normalization")
            self.gn1 = nn.Identity()
            self.gn2 = nn.Identity()
        else:
            raise ValueError(f"Invalid normalization type: {normalization_type}")

        # 2. Standard Convolutional Layers (Unchanged)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act2 = nn.SiLU()

    def forward(self, x, temb):
        residual = x
        
        # Block 1
        if self.normalization_type == "ada":
            x = self.gn1(x, temb)
        else:
            x = self.gn1(x) # Ignores temb
            
        x = self.act1(x)
        x = self.conv1(x)
        
        # Block 2
        if self.normalization_type == "ada":
            x = self.gn2(x, temb)
        else:
            x = self.gn2(x) # Ignores temb
            
        x = self.act2(x)
        x = self.conv2(x)
        
        return residual + x

class TokenDownsampler(nn.Module):
    def __init__(self, 
        in_channels, 
        feature_channels, 
        embedding_dim, 
        num_res_blocks=2, 
        learn_residual=False, 
        use_learned_components=True,
        variance_correction=None,
        normalization_type: Literal["ada", "gn", "none"] = "gn",
        naive_downsample_mode: Literal["nearest", "bilinear"] = "nearest",
    ):
        super().__init__()

        self.variance_correction = variance_correction
        self.learn_residual = learn_residual
        self.naive_downsample_mode = naive_downsample_mode
        self.use_learned_components = use_learned_components
        
        self.input_proj = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)
        # variance correction optimizable scalar for naive downsampling
        if self.variance_correction is not None:
            self.variance_correction = nn.Parameter(torch.ones(1) * self.variance_correction, requires_grad=True)
        else:
            self.variance_correction = None
        print("Downsampler variance correction: ", self.variance_correction)
        
        # 1. Learned Branch (CNN)
        self.res_blocks = []
        for _ in range(num_res_blocks):
            # Note: Residual blocks usually keep dim constant. 
            # If in_channels != out_channels, we might need a projection first, 
            # but for this specific "refinement" use-case, usually in==out.
            self.res_blocks.append(ResidualBlock(feature_channels, embedding_dim, normalization_type=normalization_type))
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
        
        if self.learn_residual:
            # ZERO-INIT the final layer of the learned branch so it starts as 0 contribution
            nn.init.zeros_(self.downsample[-1].weight)
            if self.downsample[-1].bias is not None:
                nn.init.zeros_(self.downsample[-1].bias)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, temb, foveation_mask=None):
        # Path A: Naive (Average Pooling)
        # We assume "variance preserving" average pooling often used in diffusion
        # (Avg * 2) if fitting a specific distribution, but standard Avg is safer as baseline.
        # Check if your specific VAE/DiT expects variance correction. Standard here:
        if self.learn_residual:
            if self.naive_downsample_mode == "bilinear":
                x_naive = F.avg_pool2d(x, kernel_size=2, stride=2)
            elif self.naive_downsample_mode == "nearest":
                x_naive = F.interpolate(x, scale_factor=0.5, mode="nearest")
            else:
                raise ValueError(f"Naive Downsample mode {self.naive_downsample_mode} is not supported!")
            if self.variance_correction is not None:
                x_naive = x_naive * self.variance_correction
        else:
            x_naive = 0
        
        if self.use_learned_components:
            # Path B: Learned Residual
            x = self.input_proj(x) # expand channels
            for block in self.res_blocks:
                x = block(x, temb)
            x = self.downsample(x)
        else:
            x = 0

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
        naive_upsample_mode="nearest",
        learn_residual=False,
        use_learned_components=True,
        variance_correction=None,
        normalization_type: Literal["ada", "gn", "none"] = "gn",
    ):
        super().__init__()

        self.learn_residual = learn_residual
        self.cnn_upsample_mode = cnn_upsample_mode
        self.naive_upsample_mode = naive_upsample_mode
        self.use_learned_components = use_learned_components
        self.variance_correction = variance_correction

        # variance correction optimizable scalar for naive downsampling
        if self.variance_correction is not None:
            self.variance_correction = nn.Parameter(torch.ones(1) * self.variance_correction, requires_grad=True)
        else:
            self.variance_correction = None
        print("Upsampler variance correction: ", self.variance_correction)
        
        # 1. Learned Branch (CNN)
        self.res_blocks = []
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(feature_channels, embedding_dim, normalization_type=normalization_type))
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
        if self.learn_residual:
            # ZERO-INIT the final layer so it starts as 0 contribution
            nn.init.zeros_(self.output_proj.weight)
            if self.output_proj.bias is not None:
                 nn.init.zeros_(self.output_proj.bias)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, temb, foveation_mask=None):
        
        # Path A: Naive (Bilinear Upsampling or straight-through)
        if self.learn_residual:
            if self.naive_upsample_mode == "bilinear":
                x_naive = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            elif self.naive_upsample_mode == "nearest":
                x_naive = F.interpolate(x, scale_factor=2, mode="nearest")
            else:
                raise ValueError(f"Naive Upsample mode {self.naive_upsample_mode} is not supported!")
            if self.variance_correction is not None:
                x_naive = x_naive * self.variance_correction
        else:
            x_naive = 0

        if self.use_learned_components:
            # Path B: Learned Residual
            x = self.input_proj(x)
            x = self.upsample(x)
            for block in self.res_blocks:
                x = block(x, temb)
            x = self.output_proj(x)
        else:
            x = 0

        # Output = Naive + Residual
        return x_naive + x


# ---------------------------------------------------------------------------
# DiT block-based token upsampler / downsampler (patchified domain, FLUX2)
# ---------------------------------------------------------------------------

# FLUX2 DiT constants (match flux2_dit.Flux2DiT)
_FLUX2_PATCH_TOKEN_DIM = 128  # patchified latent: 32 * 4
_FLUX2_DIT_INNER_DIM = 48 * 128  # 6144
_FLUX2_DIT_HEADS = 48
_FLUX2_DIT_HEAD_DIM = 128
_FLUX2_DIT_MLP_RATIO = 3.0
_FLUX2_ROPE_AXES_DIM = (32, 32, 32, 32)  # T, H, W, L
_FLUX2_ROPE_THETA = 2000


def _patchify_spatial_to_sequence(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Convert spatial [B, C, H, W] (C=32) to patchified sequence [B, S, D] (D=128). Same as pipeline _patchify_latents then flatten.
    Returns (seq, h, w) with S = h*w, h=H/2, w=W/2."""
    b, c, height, width = x.shape
    assert c == 32 and height % 2 == 0 and width % 2 == 0
    # 2x2 patchify: [B, 32, H, W] -> [B, 128, H/2, W/2]
    x = x.view(b, c, height // 2, 2, width // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(b, c * 4, height // 2, width // 2)
    h, w = x.shape[2], x.shape[3]
    # [B, 128, h, w] -> [B, h*w, 128]
    seq = x.flatten(2).permute(0, 2, 1)
    return seq, h, w


def _unpatchify_sequence_to_spatial(seq: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Convert sequence [B, S, D] (D=128) to spatial [B, 32, H, W]. Inverse of _patchify_spatial_to_sequence."""
    b, s, d = seq.shape
    assert d == 128 and s == h * w
    # [B, S, 128] -> [B, 128, h, w]
    x = seq.permute(0, 2, 1).view(b, d, h, w)
    # Unpatchify: [B, 128, h, w] -> [B, 32, 2*h, 2*w]
    x = x.reshape(b, 32, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(b, 32, h * 2, w * 2)
    return x


def _make_position_ids_2d(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Position ids for RoPE: [S, 4] with (T, H, W, L) = (0, row, col, 0). S = h*w."""
    rows = torch.arange(h, device=device, dtype=dtype)
    cols = torch.arange(w, device=device, dtype=dtype)
    grid_h, grid_w = torch.meshgrid(rows, cols, indexing="ij")
    t = torch.zeros_like(grid_h)
    l = torch.zeros_like(grid_h)
    ids = torch.stack([t, grid_h, grid_w, l], dim=-1)
    return ids.reshape(-1, 4)


class TokenDiTUpsampler(nn.Module):
    """
    DiT block-based token upsampler in the patchified domain (same as FLUX2 DiT tokens).
    Expands sequence by 2x in each spatial dimension (4x total): [B, S, D] -> [B, 4*S, D].
    Interface matches pipeline: forward(x, temb, foveation_mask) with x [B, 32, H, W], output [B, 32, 2*H, 2*W].
    Time conditioning is optional (temb as in flux2_image_foveated pipeline).
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_dit_blocks: int = 1,
        use_temb: bool = True,
        rope_theta: int = _FLUX2_ROPE_THETA,
        rope_axes_dim: Tuple[int, ...] = _FLUX2_ROPE_AXES_DIM,
        inner_dim: int = _FLUX2_DIT_INNER_DIM,
        num_heads: int = _FLUX2_DIT_HEADS,
        head_dim: int = _FLUX2_DIT_HEAD_DIM,
        mlp_ratio: float = _FLUX2_DIT_MLP_RATIO,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not _HAS_FLUX2_DIT:
            raise ImportError("TokenDiTUpsampler requires diffsynth.models.flux2_dit (DiffSynth-Studio).")
        assert in_channels == 32, "Patchified domain expects 32 spatial channels (128 token dim)."
        self.in_channels = in_channels
        self.token_dim = in_channels * 4  # 128
        self.use_temb = use_temb
        self.num_dit_blocks = num_dit_blocks

        self.x_embedder = nn.Linear(self.token_dim, inner_dim, bias=False)
        self.proj_out = nn.Linear(inner_dim, self.token_dim, bias=False)

        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=list(rope_axes_dim))
        if use_temb:
            self.modulation = Flux2Modulation(inner_dim, mod_param_sets=1, bias=False)
        else:
            self.modulation = None

        self.blocks = nn.ModuleList([
            Flux2SingleTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_heads,
                attention_head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                eps=eps,
                bias=False,
            )
            for _ in range(num_dit_blocks)
        ])

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None, foveation_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, 32, H, W] -> patchified sequence [B, S, 128]
        seq, h, w = _patchify_spatial_to_sequence(x)
        b, s, d = seq.shape

        # Expand sequence 2x per dim: repeat each token 4x -> [B, 4*S, 128]
        seq_exp = seq.repeat_interleave(4, dim=1)

        # Project to inner dim
        hidden = self.x_embedder(seq_exp)

        # RoPE for expanded grid (2h x 2w)
        pos_ids = _make_position_ids_2d(2 * h, 2 * w, hidden.device, hidden.dtype)
        image_rotary_emb = self.pos_embed(pos_ids)

        # Optional time modulation
        if self.modulation is not None and temb is not None:
            mod_params = self.modulation(temb)[0]
            mod_shift, mod_scale, mod_gate = mod_params
            if mod_shift.ndim == 2:
                mod_shift = mod_shift.unsqueeze(1)
                mod_scale = mod_scale.unsqueeze(1)
                mod_gate = mod_gate.unsqueeze(1)
        else:
            mod_shift = torch.zeros_like(hidden[:, :1])
            mod_scale = torch.zeros_like(hidden[:, :1])
            mod_gate = torch.ones_like(hidden[:, :1])

        for block in self.blocks:
            hidden = block(
                hidden,
                encoder_hidden_states=None,
                temb_mod_params=(mod_shift, mod_scale, mod_gate),
                image_rotary_emb=image_rotary_emb,
            )

        out = self.proj_out(hidden)
        return _unpatchify_sequence_to_spatial(out, 2 * h, 2 * w)


class TokenDiTDownsampler(nn.Module):
    """
    DiT block-based token downsampler in the patchified domain.
    Reduces sequence by 2x in each spatial dimension (1/4 total): [B, S, D] -> [B, S/4, D].
    Interface: forward(x, temb, foveation_mask) with x [B, 32, H, W], output [B, 32, H/2, W/2].
    Time conditioning is optional.
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_dit_blocks: int = 1,
        use_temb: bool = True,
        rope_theta: int = _FLUX2_ROPE_THETA,
        rope_axes_dim: Tuple[int, ...] = _FLUX2_ROPE_AXES_DIM,
        inner_dim: int = _FLUX2_DIT_INNER_DIM,
        num_heads: int = _FLUX2_DIT_HEADS,
        head_dim: int = _FLUX2_DIT_HEAD_DIM,
        mlp_ratio: float = _FLUX2_DIT_MLP_RATIO,
        eps: float = 1e-6,
        merge_mode: Literal["mean", "linear"] = "mean",
    ):
        super().__init__()
        if not _HAS_FLUX2_DIT:
            raise ImportError("TokenDiTDownsampler requires diffsynth.models.flux2_dit (DiffSynth-Studio).")
        assert in_channels == 32
        self.in_channels = in_channels
        self.token_dim = in_channels * 4
        self.use_temb = use_temb
        self.num_dit_blocks = num_dit_blocks
        self.merge_mode = merge_mode

        self.x_embedder = nn.Linear(self.token_dim, inner_dim, bias=False)
        self.proj_out = nn.Linear(inner_dim, self.token_dim, bias=False)

        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=list(rope_axes_dim))
        if use_temb:
            self.modulation = Flux2Modulation(inner_dim, mod_param_sets=1, bias=False)
        else:
            self.modulation = None

        self.blocks = nn.ModuleList([
            Flux2SingleTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_heads,
                attention_head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                eps=eps,
                bias=False,
            )
            for _ in range(num_dit_blocks)
        ])

        if merge_mode == "linear":
            self.merge = nn.Linear(self.token_dim * 4, self.token_dim, bias=False)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None, foveation_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq, h, w = _patchify_spatial_to_sequence(x)
        b, s, d = seq.shape
        assert h % 2 == 0 and w % 2 == 0, "Height and width must be even for 2x downsampling."

        hidden = self.x_embedder(seq)
        pos_ids = _make_position_ids_2d(h, w, hidden.device, hidden.dtype)
        image_rotary_emb = self.pos_embed(pos_ids)

        if self.modulation is not None and temb is not None:
            mod_params = self.modulation(temb)[0]
            mod_shift, mod_scale, mod_gate = mod_params
            if mod_shift.ndim == 2:
                mod_shift = mod_shift.unsqueeze(1)
                mod_scale = mod_scale.unsqueeze(1)
                mod_gate = mod_gate.unsqueeze(1)
        else:
            mod_shift = torch.zeros_like(hidden[:, :1])
            mod_scale = torch.zeros_like(hidden[:, :1])
            mod_gate = torch.ones_like(hidden[:, :1])

        for block in self.blocks:
            hidden = block(
                hidden,
                encoder_hidden_states=None,
                temb_mod_params=(mod_shift, mod_scale, mod_gate),
                image_rotary_emb=image_rotary_emb,
            )

        hidden = self.proj_out(hidden)
        # Merge 2x2: [B, h*w, 128] -> [B, (h/2)*(w/2), 128]
        hidden = hidden.view(b, h, w, d)
        hidden = hidden.view(b, h // 2, 2, w // 2, 2, d)
        if self.merge_mode == "mean":
            hidden = hidden.mean(dim=(2, 4))
        else:
            hidden = hidden.permute(0, 1, 3, 2, 4, 5).reshape(b, h // 2 * w // 2, d * 4)
            hidden = self.merge(hidden)
        # hidden [B, h/2, w/2, 128] -> sequence [B, (h/2)*(w/2), 128]
        seq_low = hidden.reshape(b, -1, d)
        return _unpatchify_sequence_to_spatial(seq_low, h // 2, w // 2)


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