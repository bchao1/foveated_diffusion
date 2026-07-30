# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    # apply_rotary_emb,
    # get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from .transformer_flux_base import (
    FluxAttention as BaseFluxAttention,
    FluxAttnProcessor as BaseFluxAttnProcessor,
    FluxIPAdapterAttnProcessor,
    FluxSingleTransformerBlock as BaseFluxSingleTransformerBlock,
    FluxTransformerBlock as BaseFluxTransformerBlock,
    _get_qkv_projections,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    sequence_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    cos, sin = freqs_cis  # [S, D]
    assert sequence_dim == 1
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    cos, sin = cos.to(x.device), sin.to(x.device)

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out

def get_1d_rotary_pos_embed(
    dim: int,
    theta: float = 10000.0,
    linear_factor=1.0,
    ntk_factor=1.0,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0. 
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype) / dim)) / linear_factor
    )  # [D/2]
    return freqs

def _filter_by_idx(
    tensor: torch.Tensor,
    idx: Optional[torch.Tensor],
) -> torch.Tensor:
    B, K = idx.shape
    idx_expand = idx.view(B, K, *([1] * (tensor.dim() - 2))).expand(-1, -1, *tensor.shape[2:])
    return torch.gather(tensor, dim=1, index=idx_expand)

class FluxAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        token_info: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

        mixed_res_info = token_info
        if mixed_res_info is not None:
            n_lr = mixed_res_info["n_lr"]
            rotary_emb_txt = mixed_res_info.get("rotary_txt") # ([512, 128], [...])
            rotary_emb_lr = mixed_res_info.get("rotary_lr") # ([1863, 128], [...])
            rotary_emb_hr = mixed_res_info.get("rotary_query") # ([1863, 128], [...])
            lr_overlap_idx_rev = mixed_res_info.get("lr_overlap_idx_rev")
            hr_overlap_idx_rev = mixed_res_info.get("hr_overlap_idx_rev")
            has_overlap_lr = mixed_res_info.get("has_overlap_lr", False)
            has_overlap_hr = mixed_res_info.get("has_overlap_hr", False)
            lr_rope_downsample_hr_tokens = mixed_res_info.get("lr_rope_downsample_hr_tokens", False)

            n_txt = rotary_emb_txt[0].shape[0] # 512

            # split q, k, v
            if query.shape[1] > rotary_emb_lr[0].shape[0]:
                assert attn.added_kv_proj_dim is None
                encoder_query, img_q = query[:, :n_txt], query[:, n_txt:]
                encoder_key,   img_k = key[:, :n_txt],   key[:, n_txt:]
                encoder_value, img_v = value[:, :n_txt], value[:, n_txt:]
            else:
                assert attn.added_kv_proj_dim is not None
                img_q, img_k, img_v = query, key, value

            encoder_query = apply_rotary_emb(encoder_query, rotary_emb_txt)
            encoder_key = apply_rotary_emb(encoder_key, rotary_emb_txt)

            batch_size, seq_len, _, head_dim = img_q.shape # [B, 1863, 24, 128]

            has_hr_tokens = seq_len > n_lr

            attn_outputs: List[torch.Tensor] = []

            attn_output_txt, attn_output_lr, attn_output_hr = None, None, None
            if n_lr > 0:
                assert rotary_emb_lr is not None

                cos_lr_all, sin_lr_all = rotary_emb_lr
                cos_lr_all, sin_lr_all = cos_lr_all.unsqueeze(0), sin_lr_all.unsqueeze(0)
                assert cos_lr_all.shape[1] >= n_lr

                cos_lr = cos_lr_all[:, :n_lr, ...]
                sin_lr = sin_lr_all[:, :n_lr, ...]

                key_lr = img_k[:, :n_lr, ...]
                value_lr = img_v[:, :n_lr, ...]

                key_context = key_lr
                value_context = value_lr
                cos_context = cos_lr
                sin_context = sin_lr

                if has_hr_tokens:
                    key_hr = img_k[:, n_lr:, ...]
                    value_hr = img_v[:, n_lr:, ...]
                    cos_hr = cos_lr_all[:, n_lr:, ...]
                    sin_hr = sin_lr_all[:, n_lr:, ...]

                    if has_overlap_hr:
                        key_hr = _filter_by_idx(key_hr, hr_overlap_idx_rev)
                        value_hr = _filter_by_idx(value_hr, hr_overlap_idx_rev)
                        cos_hr = _filter_by_idx(cos_hr, hr_overlap_idx_rev)
                        sin_hr = _filter_by_idx(sin_hr, hr_overlap_idx_rev)

                    if lr_rope_downsample_hr_tokens and key_hr.shape[1] > 0:
                        if key_hr.shape[1] % 4 != 0:
                            raise ValueError("lr_rope_downsample_hr_tokens requires HR tokens in 2x2 groups.")

                        num_groups = key_hr.shape[1] // 4
                        key_hr_groups = key_hr.view(batch_size, num_groups, 4, attn.heads, -1)
                        value_hr_groups = value_hr.view(batch_size, num_groups, 4, attn.heads, -1)

                        extra_dims = cos_hr.shape[2:]
                        cos_hr_groups = cos_hr.view(batch_size, num_groups, 4, *extra_dims)
                        sin_hr_groups = sin_hr.view(batch_size, num_groups, 4, *extra_dims)

                        key_hr_down = key_hr_groups[:, :, 0, ...]
                        value_hr_down = value_hr_groups[:, :, 0, ...]
                        cos_hr_down = cos_hr_groups[:, :, 0, ...]
                        sin_hr_down = sin_hr_groups[:, :, 0, ...]

                        key_context = torch.cat([key_context, key_hr_down], dim=1)
                        value_context = torch.cat([value_context, value_hr_down], dim=1)
                        cos_context = torch.cat([cos_context, cos_hr_down], dim=1)
                        sin_context = torch.cat([sin_context, sin_hr_down], dim=1)
                    else:
                        key_context = torch.cat([key_context, key_hr], dim=1)
                        value_context = torch.cat([value_context, value_hr], dim=1)
                        cos_context = torch.cat([cos_context, cos_hr], dim=1)
                        sin_context = torch.cat([sin_context, sin_hr], dim=1)

                query_lr = img_q[:, :n_lr, ...]
                query_lr = apply_rotary_emb(query_lr, (cos_lr.squeeze(0), sin_lr.squeeze(0)))
                key_context = apply_rotary_emb(key_context, (cos_context.squeeze(0), sin_context.squeeze(0)))

                # add text key, value
                key_context = torch.cat([encoder_key, key_context], dim=1)
                value_context = torch.cat([encoder_value, value_context], dim=1)

                if not has_hr_tokens:
                    # add text query
                    attn_output_txt_lr = dispatch_attention_fn(
                        torch.cat([encoder_query, query_lr], dim=1),
                        key_context, value_context,
                        attn_mask=None, backend=self._attention_backend,
                    )
                    attn_output_txt, attn_output_lr = attn_output_txt_lr[:, :encoder_query.shape[1]], attn_output_txt_lr[:, encoder_query.shape[1]:] 
                else:
                    attn_output_lr = dispatch_attention_fn(
                        query_lr,
                        key_context, value_context,
                        attn_mask=None, backend=self._attention_backend,
                    )  
                attn_outputs.append(attn_output_lr)

            if has_hr_tokens:
                assert rotary_emb_hr is not None

                cos_hr_full, sin_hr_full = rotary_emb_hr
                cos_hr_full, sin_hr_full = cos_hr_full.unsqueeze(0), sin_hr_full.unsqueeze(0)
                assert cos_hr_full.shape[1] >= seq_len

                cos_hr_tokens = cos_hr_full[:, n_lr:, ...]
                sin_hr_tokens = sin_hr_full[:, n_lr:, ...]

                query_hr = img_q[:, n_lr:, ...]
                key_hr_tokens = img_k[:, n_lr:, ...]
                value_hr_tokens = img_v[:, n_lr:, ...]

                key_hr_context = key_hr_tokens
                value_hr_context = value_hr_tokens
                cos_hr_context = cos_hr_tokens
                sin_hr_context = sin_hr_tokens

                if n_lr > 0:
                    key_lr_filtered = img_k[:, :n_lr, ...] 
                    value_lr_filtered = img_v[:, :n_lr, ...] 
                    cos_lr_filtered = cos_hr_full[:, :n_lr, ...] 
                    sin_lr_filtered = sin_hr_full[:, :n_lr, ...]

                    if has_overlap_lr:
                        key_lr_filtered = _filter_by_idx(key_lr_filtered, lr_overlap_idx_rev)
                        value_lr_filtered = _filter_by_idx(value_lr_filtered, lr_overlap_idx_rev)
                        cos_lr_filtered = _filter_by_idx(cos_lr_filtered, lr_overlap_idx_rev)
                        sin_lr_filtered = _filter_by_idx(sin_lr_filtered, lr_overlap_idx_rev)

                    key_hr_context = torch.cat([key_hr_context, key_lr_filtered], dim=1)
                    value_hr_context = torch.cat([value_hr_context, value_lr_filtered], dim=1)
                    cos_hr_context = torch.cat([cos_hr_context, cos_lr_filtered], dim=1)
                    sin_hr_context = torch.cat([sin_hr_context, sin_lr_filtered], dim=1)

                query_hr = apply_rotary_emb(query_hr, (cos_hr_tokens.squeeze(0), sin_hr_tokens.squeeze(0)))
                key_hr_context = apply_rotary_emb(key_hr_context, (cos_hr_context.squeeze(0), sin_hr_context.squeeze(0)))

                # add text key, value
                key_hr_context = torch.cat([encoder_key, key_hr_context], dim=1)
                value_hr_context = torch.cat([encoder_value, value_hr_context], dim=1)

                # add text query
                attn_output_txt_hr = dispatch_attention_fn(
                    torch.cat([encoder_query, query_hr], dim=1),
                    key_hr_context, value_hr_context,
                    attn_mask=None, backend=self._attention_backend,
                )
                attn_output_txt, attn_output_hr = attn_output_txt_hr[:, :encoder_query.shape[1]], attn_output_txt_hr[:, encoder_query.shape[1]:] 
                attn_outputs.append(attn_output_hr)
            
            if attn_output_txt is None:
                print("[WARNING] no img tokens!")
                attn_output_txt = dispatch_attention_fn(
                    encoder_query, encoder_key, encoder_value, attn_mask=None, backend=self._attention_backend,
                )

            attn_outputs = [attn_output_txt,] + attn_outputs
            hidden_states = torch.cat(attn_outputs, dim=1)
            hidden_states = hidden_states.flatten(2, 3).type_as(query)

        else:
            if attn.added_kv_proj_dim is not None:
                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)
            
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            hidden_states = dispatch_attention_fn(
                query, key, value, attn_mask=attention_mask, backend=self._attention_backend
            )
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FluxAttention(BaseFluxAttention):
    _default_processor_cls = FluxAttnProcessor
    _available_processors = [
        FluxAttnProcessor,
        FluxIPAdapterAttnProcessor,
    ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        token_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if token_info is not None:
            kwargs = dict(kwargs)
            kwargs["token_info"] = token_info
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )


@maybe_allow_in_graph
class FluxSingleTransformerBlock(BaseFluxSingleTransformerBlock):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__(dim=dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim, mlp_ratio=mlp_ratio)
        processor = self.attn.processor
        if isinstance(processor, BaseFluxAttnProcessor):
            processor = FluxAttnProcessor()
        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        token_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_info is not None:
            joint_attention_kwargs = dict(joint_attention_kwargs or {})
            joint_attention_kwargs["token_info"] = token_info
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )


@maybe_allow_in_graph
class FluxTransformerBlock(BaseFluxTransformerBlock):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6
    ):
        super().__init__(dim=dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim, qk_norm=qk_norm, eps=eps)
        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        token_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_info is not None:
            joint_attention_kwargs = dict(joint_attention_kwargs or {})
            joint_attention_kwargs["token_info"] = token_info
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(
        self, theta: int, axes_dim: List[int],
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

        freqs_dtype = torch.float64
        base_freqs = []
        for dim in self.axes_dim:
            _is_t = len(base_freqs) == 0 # t -> text
            base_freq = get_1d_rotary_pos_embed(
                dim,
                theta=self.theta,
                linear_factor=1 if _is_t else linear_factor,
                ntk_factor=1 if _is_t else ntk_factor**(dim / (dim - 2)),
                freqs_dtype=freqs_dtype,
            )
            base_freqs.append(base_freq)

        self.register_buffer("base_freqs_t", base_freqs[0], persistent=False)
        self.register_buffer("base_freqs_h", base_freqs[1], persistent=False)
        self.register_buffer("base_freqs_w", base_freqs[2], persistent=False)

    def get_rotary_for_coords(
        self,
        t_indices: torch.Tensor,
        h_indices: torch.Tensor,
        w_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns RoPE (cos, sin) embeddings for an arbitrary list of token coordinates.

        Args:
            t_indices: Tensor of shape [N] containing text indices in patch space (integer or float).
            h_indices: Tensor of shape [N] containing height indices in patch space (integer or float).
            w_indices: Tensor of shape [N] containing width indices in patch space (integer or float).

        Returns:
            Tuple of tensors (cos, sin), each of shape [N, attention_head_dim].
        """
        cos_t, sin_t = self._compute_axis_embeddings(t_indices, self.base_freqs_t)
        cos_h, sin_h = self._compute_axis_embeddings(h_indices, self.base_freqs_h)
        cos_w, sin_w = self._compute_axis_embeddings(w_indices, self.base_freqs_w)

        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)

        return cos, sin
    
    def _compute_axis_embeddings(
        self, positions: torch.Tensor, base_freqs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine and sine rotary factors for a single axis."""
        assert positions.dim() == 1

        positions = positions.to(dtype=base_freqs.dtype)
        base_freqs = base_freqs.to(device=positions.device)
        freqs = torch.outer(positions, base_freqs) # [S, D/2]
        cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() # [S, D]
        sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() # [S, D]
        return cos, sin

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        assert ids.shape[-1] == 3 
        pos = ids.float()
        freqs_cos, freqs_sin = self.get_rotary_for_coords(pos[:, 0], pos[:, 1], pos[:, 2])
        return freqs_cos.to(ids.device), freqs_sin.to(ids.device)


def idx_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    mask:   (B, N), bool
    Requirement: each batch must contain the same number of True values.
    return: (B, K)
    """
    B, N = mask.shape
    K_per_batch = mask.sum(dim=1)

    # Assert that each batch has the same K, matching the original stack assumption.
    if not torch.all(K_per_batch == K_per_batch[0]):
        raise ValueError(
            "Selected counts differ across batches, so alignment on dim=1 is not possible. "
            "Align them first or switch to padding."
        )

    K = int(K_per_batch[0].item())
    assert K != 0

    # Compute the selected column indices for each batch: (B, K)
    arange_n = torch.arange(N, device=mask.device)
    idx = arange_n.unsqueeze(0).expand(B, N)[mask].reshape(B, K)  # (B, K)
    return idx


class FluxTransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        seperate_attention: bool = True,
        lr_rope_downsample_hr_tokens: bool = True,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
        base_use_rope_lr: bool = False,
    ):
        super().__init__()
        self.seperate_attention = seperate_attention
        self.lr_rope_downsample_hr_tokens = lr_rope_downsample_hr_tokens
        self.base_use_rope_lr = base_use_rope_lr
        if lr_rope_downsample_hr_tokens:
            assert seperate_attention
        if base_use_rope_lr: 
            assert not seperate_attention

        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        self.pos_embed2 = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope, linear_factor=linear_factor, ntk_factor=ntk_factor)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self._mixed_mask_cache: Optional[Dict[str, Any]] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def _prepare_mixed_mask_state(
        self,
        mask_lr_tokens: torch.Tensor,
        mask_lr_keep_tokens: torch.Tensor,
        ht: int,
        wt: int,
    ) -> Dict[str, Any]:
        device = mask_lr_tokens.device
        batch_size = mask_lr_tokens.shape[0]

        mask_lr_keep_overlap = mask_lr_keep_tokens & mask_lr_tokens

        keep_lr = mask_lr_keep_tokens.reshape(batch_size, -1)

        y_idx = torch.arange(ht, device=device)
        x_idx = torch.arange(wt, device=device)
        Yg, Xg = torch.meshgrid(y_idx, x_idx, indexing="ij") # [32, 32] [32, 32]
        Yg_flat = Yg.reshape(-1)
        Xg_flat = Xg.reshape(-1)

        Ht2 = ht * 2
        Wt2 = wt * 2

        dy = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.long)
        dx = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.long)

        lr_linear_indices_all: List[torch.Tensor] = []
        hr_linear_indices_all: List[torch.Tensor] = []
        lr_overlap_token_flags: List[torch.Tensor] = []
        hr_overlap_token_flags: List[torch.Tensor] = []

        coords_h_lr_all: List[torch.Tensor] = []
        coords_w_lr_all: List[torch.Tensor] = []
        coords_h_hr_all: List[torch.Tensor] = []
        coords_w_hr_all: List[torch.Tensor] = []

        for bidx in range(batch_size):
            keep_lr_b = keep_lr[bidx]
            lr_idx = torch.nonzero(keep_lr_b, as_tuple=False).squeeze(1)
            lr_linear_indices_all.append(lr_idx)

            lr_overlap_flat = mask_lr_keep_overlap[bidx].reshape(-1)
            lr_overlap_flags = lr_overlap_flat.index_select(0, lr_idx).to(torch.bool)
            lr_overlap_token_flags.append(lr_overlap_flags)

            y_b = Yg_flat.index_select(0, lr_idx)
            x_b = Xg_flat.index_select(0, lr_idx)

            coords_h_lr = [y_b]
            coords_w_lr = [x_b]

            coords_h_hr = [y_b * 2]
            coords_w_hr = [x_b * 2]

            sel_hr_b = mask_lr_tokens[bidx]
            y_sel, x_sel = torch.nonzero(sel_hr_b, as_tuple=True)
            y_sel_numel = y_sel.numel()
            if y_sel_numel > 0:
                overlap_sel = mask_lr_keep_overlap[bidx][y_sel, x_sel]
                repeat_count = y_sel_numel

                y_hr_base = (2 * y_sel).repeat_interleave(4)
                x_hr_base = (2 * x_sel).repeat_interleave(4)
                y_hr_rep = y_hr_base + dy.repeat(repeat_count)
                x_hr_rep = x_hr_base + dx.repeat(repeat_count)

                hr_overlap_flags = overlap_sel.repeat_interleave(4).to(torch.bool)
                hr_overlap_token_flags.append(hr_overlap_flags)

                lin_hr = y_hr_rep * Wt2 + x_hr_rep
                hr_linear_indices_all.append(lin_hr)

                coords_h_lr.append(y_sel.repeat_interleave(4))
                coords_w_lr.append(x_sel.repeat_interleave(4))

                coords_h_hr.append(y_hr_rep)
                coords_w_hr.append(x_hr_rep)
            else:
                empty_long = torch.empty(0, dtype=torch.long, device=device)
                hr_linear_indices_all.append(empty_long)
                hr_overlap_token_flags.append(torch.empty(0, dtype=torch.bool, device=device))

            coords_h_lr_all.append(torch.cat(coords_h_lr, dim=0))
            coords_w_lr_all.append(torch.cat(coords_w_lr, dim=0))

            coords_h_hr_all.append(torch.cat(coords_h_hr, dim=0))
            coords_w_hr_all.append(torch.cat(coords_w_hr, dim=0))

        seq_lens = [coords.numel() for coords in coords_h_lr_all]
        if len(set(seq_lens)) != 1:
            raise ValueError(f"forward_mixed currently requires equal token counts across batch. Got {seq_lens}.")
        N = seq_lens[0]

        lr_lengths = [idx.numel() for idx in lr_linear_indices_all]
        if len(set(lr_lengths)) != 1:
            raise ValueError("forward_mixed expects the same num LR tokens per batch element.")
        n_lr = lr_lengths[0]

        lr_overlap_mask_tensor = torch.stack(lr_overlap_token_flags, dim=0)
        hr_overlap_mask_tensor = torch.stack(hr_overlap_token_flags, dim=0)

        # see pipeline_flux._prepare_latent_image_ids
        coords_t_lr_b = torch.zeros_like(coords_h_lr_all[0])
        coords_t_hr_b = torch.zeros_like(coords_h_hr_all[0])
        bidx = 0
        rotary_emb_lr = self.pos_embed2.get_rotary_for_coords(
            coords_t_lr_b,
            coords_h_lr_all[bidx],
            coords_w_lr_all[bidx],
        )

        rotary_emb_hr_query = self.pos_embed2.get_rotary_for_coords(
            coords_t_hr_b,
            coords_h_hr_all[bidx],
            coords_w_hr_all[bidx],
        )

        has_overlap_hr = bool(hr_overlap_mask_tensor is not None and hr_overlap_mask_tensor.numel() > 0 and hr_overlap_mask_tensor.any())
        has_overlap_lr = bool(lr_overlap_mask_tensor is not None and lr_overlap_mask_tensor.numel() > 0 and lr_overlap_mask_tensor.any())
        lr_overlap_idx_rev = idx_from_mask(~lr_overlap_mask_tensor)
        hr_overlap_idx_rev = idx_from_mask(~hr_overlap_mask_tensor)

        return {
            "mask_lr_keep_tokens": mask_lr_keep_tokens,
            "mask_lr_keep_overlap": mask_lr_keep_overlap,
            "lr_indices": tuple(lr_linear_indices_all),
            "hr_indices": tuple(hr_linear_indices_all),
            "rotary_emb_lr": rotary_emb_lr,
            "rotary_emb_hr_query": rotary_emb_hr_query,
            "lr_overlap_idx_rev": lr_overlap_idx_rev,
            "hr_overlap_idx_rev": hr_overlap_idx_rev,
            "has_overlap_hr": has_overlap_hr,
            "has_overlap_lr": has_overlap_lr,
            "n_lr": n_lr,
            "N": N,
        }

    def _get_mixed_mask_state(
        self,
        mask_lr_tokens: torch.Tensor,
        mask_lr_keep_tokens: Optional[torch.Tensor],
        batch_size: int,
        ht: int,
        wt: int,
    ) -> Dict[str, Any]:
        expected_shape = (batch_size, ht, wt)
        assert mask_lr_tokens.shape == expected_shape

        if mask_lr_keep_tokens is None:
            keep_mask = ~mask_lr_tokens
        else:
            assert mask_lr_keep_tokens.device == mask_lr_tokens.device
            assert mask_lr_keep_tokens.shape == mask_lr_tokens.shape
            assert mask_lr_keep_tokens.dtype == torch.bool
            keep_mask = mask_lr_keep_tokens

        cache = self._mixed_mask_cache
        if cache is not None:
            assert torch.equal(cache.get("mask_lr_tokens"), mask_lr_tokens)
            assert torch.equal(cache.get("mask_lr_keep_tokens"), keep_mask)
            return cache["data"]

        mask_clone = mask_lr_tokens.detach().clone()
        keep_clone = keep_mask.detach().clone()
        mask_state = self._prepare_mixed_mask_state(mask_clone, keep_mask, ht, wt)
        self._mixed_mask_cache = {
            "mask_lr_tokens": mask_clone,
            "mask_lr_keep_tokens": keep_clone,
            "data": mask_state,
        }
        return mask_state
        
    def forward_mixed(
        self,
        hidden_states_lr: torch.Tensor,
        hidden_states_hr: torch.Tensor,
        mask_lr_tokens: torch.Tensor = None,
        mask_lr_keep_tokens: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids_lr: torch.Tensor = None,
        img_ids_hr: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Mixed-resolution forward path.

        - Uses `mask_lr_tokens` (shape [B, 1, Ht, Wt] in patch space) to pick which HR sub-tokens (2x2 per selected
          LR token) to include from the HR latent. When `mask_lr_keep_tokens` is provided, it describes which LR tokens
          are kept for the LR branch, allowing overlap with the HR area.
        - Returns two noise predictions: for LR and HR latents respectively, both in 5D grids matching the inputs.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        
        b    = hidden_states_lr.shape[0] # [B, 1024, 64]
        b_hr = hidden_states_hr.shape[0] # [B, 4096, 64]
        b_m, _, ht, wt = mask_lr_tokens.shape # [B, 1, 32, 32] 
        assert b == b_hr and b == b_m 

        # Compute linear-embedded tokens for LR and HR
        tokens_lr_flat = self.x_embedder(hidden_states_lr) # [B, 1024, 3072]
        tokens_hr_flat = self.x_embedder(hidden_states_hr) # [B, 4096, 3072]

        # -----------------------------------------
        mask_state = self._get_mixed_mask_state(mask_lr_tokens.squeeze(1), mask_lr_keep_tokens.squeeze(1), b, ht, wt)

        lr_indices = mask_state["lr_indices"]
        hr_indices = mask_state["hr_indices"]
        N = mask_state["N"]
        n_lr = mask_state["n_lr"]

        mixed_tokens_list: List[torch.Tensor] = []
        for bidx in range(b):
            lr_tokens_b = tokens_lr_flat[bidx].index_select(0, lr_indices[bidx])
            hr_idx = hr_indices[bidx]
            if n_lr < N:
                hr_tokens_b = tokens_hr_flat[bidx].index_select(0, hr_idx)
                lr_tokens_b = torch.cat([lr_tokens_b, hr_tokens_b], dim=0)
            if lr_tokens_b.shape[0] != N:
                raise RuntimeError("Mixed token assembly produced inconsistent sequence length.")
            mixed_tokens_list.append(lr_tokens_b)

        mixed_tokens = torch.stack(mixed_tokens_list, dim=0) # [B, N, D]

        rotary_emb_lr = mask_state["rotary_emb_lr"]
        rotary_emb_hr_query = mask_state["rotary_emb_hr_query"]
        lr_overlap_idx_rev = mask_state["lr_overlap_idx_rev"]
        hr_overlap_idx_rev = mask_state["hr_overlap_idx_rev"]
        has_overlap_hr = mask_state["has_overlap_hr"]
        has_overlap_lr = mask_state["has_overlap_lr"]
        # ---- 

        timestep = timestep.to(hidden_states_lr.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states_lr.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        assert txt_ids.ndim == 2
        # ids = torch.cat((txt_ids, img_ids), dim=0) # [512,3] + [1024,3]
        # image_rotary_emb = self.pos_embed(ids) # ([1536, 128], [1536, 128])
        # rotary_emb_lr # ([1863, 128], [1863, 128])
        # rotary_emb_hr_query # ([1863, 128], [1863, 128])
        rotary_txt = self.pos_embed(txt_ids)

        assert 'ip_adapter_image_embeds' not in joint_attention_kwargs

        # ---------------------------------------
        # Transformer blocks over mixed tokens
        hidden_states = mixed_tokens  # [B, N, D]

        if n_lr == N or not self.seperate_attention:
            token_info = None
            if self.base_use_rope_lr:
                image_rotary_emb = (torch.cat([rotary_txt[0], rotary_emb_lr[0]]), torch.cat([rotary_txt[1], rotary_emb_lr[1]]))
            else:
                image_rotary_emb = (torch.cat([rotary_txt[0], rotary_emb_hr_query[0]]), torch.cat([rotary_txt[1], rotary_emb_hr_query[1]]))
        else:
            token_info = {
                "n_lr": n_lr,
                "rotary_txt": rotary_txt,
                "rotary_lr": rotary_emb_lr,
                "rotary_query": rotary_emb_hr_query,
                "lr_overlap_idx_rev": lr_overlap_idx_rev,
                "hr_overlap_idx_rev": hr_overlap_idx_rev,
                "has_overlap_lr": has_overlap_lr,
                "has_overlap_hr": has_overlap_hr,
                "lr_rope_downsample_hr_tokens": getattr(self.config, "lr_rope_downsample_hr_tokens", False),
            }
            image_rotary_emb = None

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                token_info=token_info,
            )

        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                token_info=token_info,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states) # [B, 1863, 64]

        new_lr = hidden_states_lr.clone()
        new_hr = hidden_states_hr.clone()
        for bidx in range(b):
            new_lr[bidx].index_copy_(0, lr_indices[bidx], hidden_states[bidx, :n_lr])
            if n_lr < N:
                new_hr[bidx].index_copy_(0, hr_indices[bidx], hidden_states[bidx, n_lr:])

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        return new_lr, new_hr
