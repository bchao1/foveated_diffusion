import os 
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch 
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_file

from diffusers.models.activations import get_activation

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):           # x: (N, C, H, W)
        x = x.permute(0, 2, 3, 1)   # (N, H, W, C)
        x = self.ln(x)              # norm over C
        x = x.permute(0, 3, 1, 2)   # back to (N, C, H, W)
        return x

class SimpleResidualBlock2D(nn.Module):
    r"""
    A simple residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        dropout (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        groups: int = 32,
        eps: float = 1e-6,
        norm_type: str = "layer_norm",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        # layers
        self.norm_type = norm_type
        if norm_type == "layer_norm":
            self.norm1 = LayerNorm2d(in_dim, eps=eps, affine=True)
            self.norm2 = LayerNorm2d(out_dim, eps=eps, affine=True)
        else:
            self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_dim, eps=eps, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_dim, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)
        # Dropout
        x = self.dropout(x)
        x = self.conv2(x)

        # Add residual connection
        return x + h


def _parse_scale(scale: float | Sequence[float]) -> Tuple[float, float]:
    if isinstance(scale, (int, float)):
        value = float(scale)
        return value, value
    if len(scale) != 2:
        raise ValueError(f"scale must be a float or sequence of length 2, got {scale}")
    return float(scale[0]), float(scale[1])


class FluxLatentResizer(nn.Module):
    """Simple ResNet based latent resizer operating on 2D feature maps."""

    def __init__(
        self,
        channels: int = 16,
        hidden_dim: int = 128,
        num_res_blocks: int = 20,
        scale: float | Sequence[float] = 1.0,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        groups: int = 32,
        eps: float = 1e-6,
        interpolate_mode: str = "nearest",
        interpolate_align_corners: bool = False,
        resizer_type: str | None = None,
        norm_type: str = "layer_norm",
    ) -> None:
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1")

        self.channels = channels
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks
        self.interpolate_mode = interpolate_mode
        self.interpolate_align_corners = interpolate_align_corners
        self.scale = _parse_scale(scale)
        self.resizer_type = resizer_type

        self.conv_in = nn.Conv2d(channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.pre_block = SimpleResidualBlock2D(
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            non_linearity=non_linearity,
            groups=groups,
            eps=eps,
            norm_type=norm_type,
        )
        post_blocks = []
        for _ in range(num_res_blocks - 1):
            post_blocks.append(
                SimpleResidualBlock2D(
                    hidden_dim,
                    hidden_dim,
                    dropout=dropout,
                    non_linearity=non_linearity,
                    groups=groups,
                    eps=eps,
                    norm_type=norm_type,
                )
            )
        self.post_blocks = nn.ModuleList(post_blocks)
        self.conv_out = nn.Conv2d(hidden_dim, channels, kernel_size=3, stride=1, padding=1)
        
        if self.scale == (0.5, 0.5):
            self.down_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.pre_block(h)

        if self.scale != (1.0, 1.0):
            if self.scale == (0.5, 0.5):
                h = self.down_conv(h)
            else:
                interpolate_kwargs = {"mode": self.interpolate_mode}
                if self.interpolate_mode in {"bilinear", "bicubic", "trilinear", "linear"}:
                    interpolate_kwargs["align_corners"] = self.interpolate_align_corners
                h = F.interpolate(h, scale_factor=self.scale, recompute_scale_factor=False, **interpolate_kwargs)

        for block in self.post_blocks:
            h = block(h)

        h = self.conv_out(h)
        return h

def load_resizer_state(model: nn.Module, state_path: Path) -> None:
    if not state_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {state_path}")
    state_dict = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

def load_checkpoint(
    model: nn.Module,
    path: str,
    device_map: torch.device | str = "cpu",
    strict: bool = True,
) -> nn.Module:
    """
    Loads a checkpoint into model from a path. Support three types of checkpoints:
        1. huggingface safetensors
        2. local .pt or .pth

    Returns:
        nn.Module: The model with the loaded checkpoint.
    """
    assert os.path.exists(path), f"Could not find checkpoint at {path}"

    if path.endswith(".safetensors"):
        ckpt = load_file(path, device='cpu')
        missing, unexpected = model.load_state_dict(ckpt, strict=strict)
        # print(missing, unexpected)
    elif path.endswith(".pt") or path.endswith(".pth"):
        ckpt = torch.load(path, map_location=device_map)
        missing, unexpected = model.load_state_dict(ckpt, strict=strict)
        # print(missing, unexpected)
    else:
        raise NotImplementedError
    return model

def instantiate_resizer(
    ckpt_path: str,
    resizer_type: str,
    norm_type: str,
    device: torch.device,
    dtype: torch.dtype,
) -> FluxLatentResizer:
    scale = (0.5, 0.5) if resizer_type in {"down", "downscaler", "downscale"} else (2.0, 2.0)
    model = FluxLatentResizer(
        scale=scale,
        resizer_type=resizer_type,
        norm_type=norm_type,
    )

    if ckpt_path is not None:
        checkpoint_path = Path(ckpt_path)
        if checkpoint_path.is_dir():
            # checkpoint_path = checkpoint_path / "model_state.pt"
            checkpoint_path = checkpoint_path / "ema.safetensor"
        # load_resizer_state(model, checkpoint_path)
        model = load_checkpoint(model, str(checkpoint_path))
    else:
        print("[WARNING] random weights for resizer!")

    model = model.to(device=device, dtype=dtype)

    model.eval()

    return model 