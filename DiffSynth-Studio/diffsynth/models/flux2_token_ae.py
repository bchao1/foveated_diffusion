# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Token upsampler and downsampler for FLUX2 VAE latents.

Compresses latents (B, C, H, W) by 2X spatially and upsamples by 2X using the same
architectural choices as flux2_vae.py: ResnetBlock2D, Downsample2D, Upsample2D.
"""

from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flux2_vae import ResnetBlock2D, Downsample2D, Upsample2D, get_activation, Attention


class Flux2TokenDownsampler(nn.Module):
    """
    Downsample latents (B, C, H, W) -> (B, C, H/2, W/2) by 2X.

    Learned flux2_vae Encoder-style architecture:
    - conv_in: project C -> 4C (residual blocks operate at 4x channels when spatial res is halved)
    - ResnetBlock2D at 4C channels
    - Downsample2D with use_conv=True, stride=2, padding=0
    - conv_out: project 4C -> C

    If naive_downsample_mode is "nearest" or "bilinear", adds residual path:
    output = learned_output + naive_downsample(input) (network learns residual of naive downsampling)
    """

    @classmethod
    def from_config(cls, config: dict) -> "Flux2TokenDownsampler":
        """Create from config dict (e.g. from tokenAE_config.yaml)."""
        return cls(
            in_channels=config.get("in_channels", 32),
            num_res_blocks=config.get("downsampler_num_res_blocks", 2),
            resnet_eps=config.get("flux2_resnet_eps", 1e-6),
            resnet_act_fn=config.get("flux2_resnet_act_fn", "swish"),
            resnet_groups=config.get("flux2_resnet_groups", 32),
            naive_downsample_mode=config.get("flux2_naive_downsample_mode"),
            add_norm_act_before_conv_out=config.get("flux2_downsampler_add_norm_act_before_conv_out", True),
            add_attention=config.get("flux2_downsampler_add_attention", False),
            attention_head_dim=config.get("flux2_downsampler_attention_head_dim"),
            attn_groups=config.get("flux2_downsampler_attn_groups"),
        )

    def __init__(
        self,
        in_channels: int,
        num_res_blocks: int = 2,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        naive_downsample_mode: Optional[Literal["nearest", "bilinear"]] = None,
        add_norm_act_before_conv_out: bool = True,
        add_attention: bool = False,
        attention_head_dim: Optional[int] = None,
        attn_groups: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.naive_downsample_mode = naive_downsample_mode
        self.add_attention = add_attention
        block_channels = in_channels * 4
        self.conv_in = nn.Conv2d(in_channels, block_channels, kernel_size=3, stride=1, padding=1)

        attn_groups = attn_groups if attn_groups is not None else resnet_groups
        if attention_head_dim is None:
            attention_head_dim = block_channels

        resnets = []
        attentions = []
        output_scale_factor = 1.0

        if add_attention:
            resnets.append(
                ResnetBlock2D(
                    in_channels=block_channels,
                    out_channels=block_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=0.0,
                    time_embedding_norm="default",
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=True,
                )
            )
            for _ in range(num_res_blocks):
                attentions.append(
                    Attention(
                        block_channels,
                        heads=block_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
                resnets.append(
                    ResnetBlock2D(
                        in_channels=block_channels,
                        out_channels=block_channels,
                        temb_channels=None,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=0.0,
                        time_embedding_norm="default",
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=True,
                    )
                )
        else:
            for _ in range(num_res_blocks):
                resnets.append(
                    ResnetBlock2D(
                        in_channels=block_channels,
                        out_channels=block_channels,
                        temb_channels=None,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=0.0,
                        time_embedding_norm="default",
                        non_linearity=resnet_act_fn,
                        output_scale_factor=1.0,
                        pre_norm=True,
                    )
                )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions) if attentions else nn.ModuleList()
        self.downsample = Downsample2D(
            block_channels,
            use_conv=True,
            out_channels=block_channels,
            padding=0,
            name="op",
        )
        self.add_norm_act_before_conv_out = add_norm_act_before_conv_out
        if add_norm_act_before_conv_out:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_channels, num_groups=resnet_groups, eps=resnet_eps)
            self.conv_act = get_activation(resnet_act_fn)
        self.conv_out = nn.Conv2d(block_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        temb = None
        residual_high_res = hidden_states if self.naive_downsample_mode is not None else None
        hidden_states = self.conv_in(hidden_states)
        if self.add_attention:
            hidden_states = self.resnets[0](hidden_states, temb=temb)
            for attn, resnet in zip(self.attentions, self.resnets[1:]):
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = resnet(hidden_states, temb=temb)
        else:
            for resnet in self.resnets:
                hidden_states = resnet(hidden_states, temb=temb)
        hidden_states = self.downsample(hidden_states)
        if self.add_norm_act_before_conv_out:
            hidden_states = self.conv_norm_out(hidden_states)
            hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        if residual_high_res is not None:
            naive_down = F.interpolate(residual_high_res, scale_factor=0.5, mode=self.naive_downsample_mode)
            hidden_states = hidden_states + naive_down
        return hidden_states


class Flux2TokenUpsampler(nn.Module):
    """
    Upsample latents (B, C, h, w) -> (B, C, 2h, 2w) by 2X.

    - conv_in: project C -> 4C
    - [Optional] pre_upsampler_res_blocks: process low-res, residual add to conv_in output
    - Upsample2D (spatial res doubles)
    - ResnetBlock2D at 4C channels
    - conv_out: project 4C -> C
    - [Optional] naive_upsample_mode: when "nearest" or "bilinear", adds residual path:
      output = naive_upsample(input) + network_output (network learns residual of naive upsampling)
    """

    @classmethod
    def from_config(cls, config: dict) -> "Flux2TokenUpsampler":
        """Create from config dict (e.g. from tokenAE_config.yaml)."""
        return cls(
            in_channels=config.get("in_channels", 32),
            num_res_blocks=config.get("upsampler_num_res_blocks", 3),
            num_pre_upsampler_res_blocks=config.get("flux2_upsampler_num_pre_upsampler_res_blocks", 0),
            resnet_eps=config.get("flux2_resnet_eps", 1e-6),
            resnet_act_fn=config.get("flux2_resnet_act_fn", "swish"),
            resnet_groups=config.get("flux2_resnet_groups", 32),
            add_norm_act_before_conv_out=config.get("flux2_upsampler_add_norm_act_before_conv_out", True),
            add_attention=config.get("flux2_upsampler_add_attention", False),
            attention_head_dim=config.get("flux2_upsampler_attention_head_dim"),
            attn_groups=config.get("flux2_upsampler_attn_groups"),
            naive_upsample_mode=config.get("flux2_upsampler_naive_upsample_mode"),
        )

    def __init__(
        self,
        in_channels: int,
        num_res_blocks: int = 3,
        num_pre_upsampler_res_blocks: int = 0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_norm_act_before_conv_out: bool = True,
        add_attention: bool = False,
        attention_head_dim: Optional[int] = None,
        attn_groups: Optional[int] = None,
        naive_upsample_mode: Optional[Literal["nearest", "bilinear"]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.naive_upsample_mode = naive_upsample_mode
        block_channels = in_channels * 4
        self.conv_in = nn.Conv2d(in_channels, block_channels, kernel_size=3, stride=1, padding=1)
        # Pre-upsampler res blocks: process low-res, residual add before upsample
        self.pre_upsampler_resnets = nn.ModuleList()
        for _ in range(num_pre_upsampler_res_blocks):
            self.pre_upsampler_resnets.append(
                ResnetBlock2D(
                    in_channels=block_channels,
                    out_channels=block_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=0.0,
                    time_embedding_norm="default",
                    non_linearity=resnet_act_fn,
                    output_scale_factor=1.0,
                    pre_norm=True,
                )
            )
        self.upsample = Upsample2D(
            block_channels,
            use_conv=True,
            out_channels=block_channels,
            name="conv",
            padding=1,
            interpolate=True,
        )

        self.add_attention = add_attention
        attn_groups = attn_groups if attn_groups is not None else resnet_groups
        if attention_head_dim is None:
            attention_head_dim = block_channels
        output_scale_factor = 1.0

        resnets = []
        attentions = []
        if add_attention:
            resnets.append(
                ResnetBlock2D(
                    in_channels=block_channels,
                    out_channels=block_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=0.0,
                    time_embedding_norm="default",
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=True,
                )
            )
            for _ in range(num_res_blocks):
                attentions.append(
                    Attention(
                        block_channels,
                        heads=block_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
                resnets.append(
                    ResnetBlock2D(
                        in_channels=block_channels,
                        out_channels=block_channels,
                        temb_channels=None,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=0.0,
                        time_embedding_norm="default",
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=True,
                    )
                )
        else:
            for _ in range(num_res_blocks):
                resnets.append(
                    ResnetBlock2D(
                        in_channels=block_channels,
                        out_channels=block_channels,
                        temb_channels=None,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=0.0,
                        time_embedding_norm="default",
                        non_linearity=resnet_act_fn,
                        output_scale_factor=1.0,
                        pre_norm=True,
                    )
                )
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions) if attentions else nn.ModuleList()
        self.add_norm_act_before_conv_out = add_norm_act_before_conv_out
        if add_norm_act_before_conv_out:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_channels, num_groups=resnet_groups, eps=resnet_eps)
            self.conv_act = get_activation(resnet_act_fn)
        self.conv_out = nn.Conv2d(block_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        temb = None
        residual_low_res = hidden_states if self.naive_upsample_mode is not None else None
        hidden_states = self.conv_in(hidden_states)
        # Pre-upsampler: residual connection (low_res + processed)
        if len(self.pre_upsampler_resnets) > 0:
            residual = hidden_states
            for resnet in self.pre_upsampler_resnets:
                hidden_states = resnet(hidden_states, temb=temb)
            hidden_states = hidden_states + residual
        hidden_states = self.upsample(hidden_states)
        if self.add_attention:
            hidden_states = self.resnets[0](hidden_states, temb=temb)
            for attn, resnet in zip(self.attentions, self.resnets[1:]):
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = resnet(hidden_states, temb=temb)
        else:
            for resnet in self.resnets:
                hidden_states = resnet(hidden_states, temb=temb)
        if self.add_norm_act_before_conv_out:
            hidden_states = self.conv_norm_out(hidden_states)
            hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        if residual_low_res is not None:
            naive_up = F.interpolate(residual_low_res, scale_factor=2, mode=self.naive_upsample_mode)
            hidden_states = naive_up + hidden_states
        return hidden_states
