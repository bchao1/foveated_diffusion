from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.autoencoders.autoencoder_kl_wan import (
    AvgDown3D,
    AutoencoderKLWan,
    DupUp3D,
    WanCausalConv3d,
    WanResidualBlock,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


def _infer_encoder_mid_dim(vae: AutoencoderKLWan) -> int:
    mid_block = vae.encoder.mid_block
    if hasattr(mid_block, "dim"):
        return int(mid_block.dim)

    resnets = getattr(mid_block, "resnets", None)
    if isinstance(resnets, nn.ModuleList) and len(resnets) > 0:
        first = resnets[0]
        if hasattr(first, "out_dim"):
            return int(first.out_dim)
        if hasattr(first, "conv1"):
            return int(first.conv1.in_channels)

    raise RuntimeError("Unable to infer encoder mid-block feature dimension from the Wan VAE.")


def prepare_latent_stats(
    vae: AutoencoderKLWan,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(vae.config.latents_mean, dtype=dtype).view(1, vae.config.z_dim, 1, 1, 1)
    inv_std = (
        1.0
        / torch.as_tensor(vae.config.latents_std, dtype=dtype).view(1, vae.config.z_dim, 1, 1, 1)
    )
    return mean, inv_std


class WanLatentFeatureSpatialUpscaler(nn.Module):
    def __init__(
        self,
        *,
        vae: AutoencoderKLWan,
        scale: tuple[int, int, int] = (1, 2, 2),
        num_res_blocks: int = 3,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        assert scale == (1, 2, 2)
        assert num_res_blocks > 1

        vae_param = next(vae.parameters())
        self._vae_dtype = vae_param.dtype
        self._bridge_dtype = dtype

        self.post_quant_conv = copy.deepcopy(vae.post_quant_conv)
        self.decoder_conv_in = copy.deepcopy(vae.decoder.conv_in)
        self.encoder_norm_out = copy.deepcopy(vae.encoder.norm_out)
        self.encoder_nonlinearity = copy.deepcopy(vae.encoder.nonlinearity)
        self.encoder_conv_out = copy.deepcopy(vae.encoder.conv_out)
        self.quant_conv = copy.deepcopy(vae.quant_conv)

        decoder_dim = int(vae.decoder.conv_in.out_channels)
        encoder_dim = _infer_encoder_mid_dim(vae)

        self.bridge_conv_in = WanCausalConv3d(decoder_dim, encoder_dim, kernel_size=1)
        self.spatial_upsample = DupUp3D(encoder_dim, encoder_dim, factor_t=1, factor_s=scale[1])
        self.res_blocks = nn.ModuleList(
            [WanResidualBlock(encoder_dim, encoder_dim, dropout, non_linearity) for _ in range(num_res_blocks)]
        )
        self.post_resize_conv = nn.Conv2d(encoder_dim, encoder_dim, kernel_size=3, stride=1, padding=1)
        self.bridge_conv_in = self.bridge_conv_in.to(dtype=self._bridge_dtype)
        self.spatial_upsample = self.spatial_upsample.to(dtype=self._bridge_dtype)
        self.res_blocks = self.res_blocks.to(dtype=self._bridge_dtype)
        self.post_resize_conv = self.post_resize_conv.to(dtype=self._bridge_dtype)

        mean, inv_std = prepare_latent_stats(vae, dtype=self._vae_dtype)
        self.register_buffer("_latents_mean", mean, persistent=False)
        self.register_buffer("_latents_inv_std", inv_std, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        latent_norm: torch.Tensor,
    ):
        input_dtype = latent_norm.dtype
        latent_fp = latent_norm.to(dtype=self._vae_dtype)
        latents_mean = self._latents_mean
        latents_inv_std = self._latents_inv_std

        latent_raw = latent_fp / latents_inv_std + latents_mean
        x = self.post_quant_conv(latent_raw)
        x = self.decoder_conv_in(x)

        feature = x.to(dtype=self._bridge_dtype)
        feature = self.bridge_conv_in(feature)

        resize_shortcut = self.spatial_upsample(feature)
        b, c, t, h, w = resize_shortcut.shape
        feature_2d = resize_shortcut.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        feature_2d = self.post_resize_conv(feature_2d)
        feature = feature_2d.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)

        for block in self.res_blocks:
            feature = block(feature)
        feature = feature + resize_shortcut
        feature_pred = feature.to(dtype=self._vae_dtype)

        x = self.encoder_norm_out(feature_pred)
        x = self.encoder_nonlinearity(x)
        x = self.encoder_conv_out(x)
        x = self.quant_conv(x)

        posterior = DiagonalGaussianDistribution(x)
        latent = posterior.sample()
        latent_norm = (latent - latents_mean) * latents_inv_std
        latent_norm = latent_norm.to(dtype=input_dtype)

        return latent_norm


class WanLatentFeatureSpatialDownscaler(nn.Module):
    def __init__(
        self,
        *,
        vae: AutoencoderKLWan,
        scale: tuple[float, float, float] = (1.0, 0.5, 0.5),
        num_res_blocks: int = 3,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        assert scale == (1, 0.5, 0.5)
        assert num_res_blocks > 1

        factor_s = 2

        vae_param = next(vae.parameters())
        self._vae_dtype = vae_param.dtype
        self._bridge_dtype = dtype

        self.post_quant_conv = copy.deepcopy(vae.post_quant_conv)
        self.decoder_conv_in = copy.deepcopy(vae.decoder.conv_in)
        self.encoder_norm_out = copy.deepcopy(vae.encoder.norm_out)
        self.encoder_nonlinearity = copy.deepcopy(vae.encoder.nonlinearity)
        self.encoder_conv_out = copy.deepcopy(vae.encoder.conv_out)
        self.quant_conv = copy.deepcopy(vae.quant_conv)

        decoder_dim = int(vae.decoder.conv_in.out_channels)
        encoder_dim = _infer_encoder_mid_dim(vae)

        self.bridge_conv_in = WanCausalConv3d(decoder_dim, encoder_dim, kernel_size=1)
        self.spatial_downsample = AvgDown3D(encoder_dim, encoder_dim, factor_t=1, factor_s=factor_s)
        self.conv_down = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(encoder_dim, encoder_dim, kernel_size=3, stride=(factor_s, factor_s)),
        )
        self.res_blocks = nn.ModuleList(
            [WanResidualBlock(encoder_dim, encoder_dim, dropout, non_linearity) for _ in range(num_res_blocks)]
        )
        self.bridge_conv_in = self.bridge_conv_in.to(dtype=self._bridge_dtype)
        self.spatial_downsample = self.spatial_downsample.to(dtype=self._bridge_dtype)
        self.conv_down = self.conv_down.to(dtype=self._bridge_dtype)
        self.res_blocks = self.res_blocks.to(dtype=self._bridge_dtype)

        mean, inv_std = prepare_latent_stats(vae, dtype=self._vae_dtype)
        self.register_buffer("_latents_mean", mean, persistent=False)
        self.register_buffer("_latents_inv_std", inv_std, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        latent_norm: torch.Tensor,
    ):
        input_dtype = latent_norm.dtype
        latent_fp = latent_norm.to(dtype=self._vae_dtype)
        latents_mean = self._latents_mean
        latents_inv_std = self._latents_inv_std

        latent_raw = latent_fp / latents_inv_std + latents_mean
        x = self.post_quant_conv(latent_raw)
        x = self.decoder_conv_in(x)

        feature = x.to(dtype=self._bridge_dtype)
        feature = self.bridge_conv_in(feature)

        resize_shortcut = self.spatial_downsample(feature)
        b, c, t, h, w = feature.shape
        feature_2d = feature.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        feature_2d = self.conv_down(feature_2d)
        feature = feature_2d.view(b, t, feature_2d.size(1), feature_2d.size(2), feature_2d.size(3)).permute(0, 2, 1, 3, 4)

        for block in self.res_blocks:
            feature = block(feature)
        feature = feature + resize_shortcut
        feature_pred = feature.to(dtype=self._vae_dtype)

        x = self.encoder_norm_out(feature_pred)
        x = self.encoder_nonlinearity(x)
        x = self.encoder_conv_out(x)
        x = self.quant_conv(x)

        posterior = DiagonalGaussianDistribution(x)
        latent = posterior.sample()
        latent_norm = (latent - latents_mean) * latents_inv_std
        latent_norm = latent_norm.to(dtype=input_dtype)

        return latent_norm


def load_wan_resizer(
    *,
    vae: AutoencoderKLWan,
    mode: str,
    checkpoint_path: str,
    num_res_blocks: int = 3,
    dropout: float = 0.0,
    non_linearity: str = "silu",
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    assert device is not None
    assert dtype is not None
    assert mode in {"up_s", "down_s"}

    device = torch.device(device)

    if mode == "up_s":
        resizer: nn.Module = WanLatentFeatureSpatialUpscaler(
            vae=vae,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            non_linearity=non_linearity,
            dtype=dtype,
        )
    else:
        resizer = WanLatentFeatureSpatialDownscaler(
            vae=vae,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            non_linearity=non_linearity,
            dtype=dtype,
        )

    state_path = Path(checkpoint_path)
    if state_path.is_dir():
        state_path = state_path / "model_state.pt"
    state_dict = torch.load(state_path, map_location="cpu")
    incompatible = resizer.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        "post_quant_conv.",
        "decoder_conv_in.",
        "encoder_norm_out.",
        "encoder_nonlinearity.",
        "encoder_conv_out.",
        "quant_conv.",
    )
    invalid_missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_missing_prefixes)
    ]
    if incompatible.unexpected_keys or invalid_missing:
        raise RuntimeError(
            "Failed to load bridge checkpoint cleanly. "
            f"unexpected_keys={incompatible.unexpected_keys}, invalid_missing_keys={invalid_missing}"
        )
    resizer = resizer.to(device=device)
    resizer = resizer.eval()

    return resizer