# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

import torch
import torch.nn as nn

from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_dispatch import dispatch_attention_fn

from .transformer_wan_base import (
    WanAttention as BaseWanAttention,
    WanAttnProcessor as BaseWanAttnProcessor,
    WanTransformerBlock as BaseWanTransformerBlock,
    WanTransformer3DModel as BaseWanTransformer3DModel,
    _get_added_kv_projections,
    _get_qkv_projections,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def get_1d_rotary_pos_embed(
    dim: int,
    theta: float = 10000.0,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    freqs_dtype: torch.dtype = torch.float32,  # torch.float32, torch.float64 (flux)
):
    """
    Compute the base frequency vector for sinusoidal rotary embeddings.

    Args:
        dim (`int`): Even dimensionality of the rotary embedding block.
        theta (`float`, *optional*, defaults to 10000.0): Scaling factor for frequency computation.
        linear_factor (`float`, *optional*, defaults to 1.0): Context extrapolation scaling factor.
        ntk_factor (`float`, *optional*, defaults to 1.0): NTK-Aware RoPE scaling factor.
        freqs_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`): Dtype of the returned base frequencies.

    Returns:
        `torch.Tensor`: Base frequencies of shape `[dim // 2]`.
    """
    assert dim % 2 == 0
    theta = theta * ntk_factor
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype) / dim)) # [D/2]
    return freqs / linear_factor

def _filter_by_idx(
    tensor: torch.Tensor,
    idx: Optional[torch.Tensor],
) -> torch.Tensor:
    B, K = idx.shape
    idx_expand = idx.view(B, K, *([1] * (tensor.dim() - 2))).expand(-1, -1, *tensor.shape[2:])
    return torch.gather(tensor, dim=1, index=idx_expand)


class WanAttnProcessor(BaseWanAttnProcessor):
    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        token_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if token_info is None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        def apply_rotary_emb(
            hidden_states_tensor: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
        ) -> torch.Tensor:
            x1, x2 = hidden_states_tensor.unflatten(-1, (-1, 2)).unbind(-1)
            cos = freqs_cos[..., 0::2]
            sin = freqs_sin[..., 1::2]
            out = torch.empty_like(hidden_states_tensor)
            out[..., 0::2] = x1 * cos - x2 * sin
            out[..., 1::2] = x1 * sin + x2 * cos
            return out.type_as(hidden_states_tensor)

        mixed_res_info = token_info

        if encoder_hidden_states_img is not None:
            raise ValueError("Mixed-resolution attention is not supported with added KV projections.")
        n_lr = mixed_res_info["n_lr"]
        rotary_emb_hr = mixed_res_info.get("rotary_query")
        lr_overlap_idx_rev = mixed_res_info.get("lr_overlap_idx_rev")
        hr_overlap_idx_rev = mixed_res_info.get("hr_overlap_idx_rev")
        has_overlap_lr = mixed_res_info.get("has_overlap_lr", False)
        has_overlap_hr = mixed_res_info.get("has_overlap_hr", False)
        lr_rope_downsample_hr_tokens = mixed_res_info.get("lr_rope_downsample_hr_tokens", False)

        batch_size, seq_len, _, _ = query.shape
        if n_lr < 0 or n_lr > seq_len:
            raise ValueError(f"token_info n_lr out of range: {n_lr} vs seq_len={seq_len}")

        has_hr_tokens = seq_len > n_lr

        attn_outputs: List[torch.Tensor] = []

        if n_lr > 0:
            if rotary_emb is None:
                raise ValueError("Expected rotary_emb when performing LR mixed-resolution attention.")

            cos_lr_all, sin_lr_all = rotary_emb
            if cos_lr_all.shape[1] < n_lr:
                raise ValueError("Rotary embedding for LR tokens has insufficient length.")

            cos_lr = cos_lr_all[:, :n_lr, ...]
            sin_lr = sin_lr_all[:, :n_lr, ...]

            key_lr = key[:, :n_lr, ...]
            value_lr = value[:, :n_lr, ...]

            key_context = key_lr
            value_context = value_lr
            cos_context = cos_lr
            sin_context = sin_lr

            if has_hr_tokens:
                key_hr = key[:, n_lr:, ...]
                value_hr = value[:, n_lr:, ...]
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

            query_lr = query[:, :n_lr, ...]
            query_lr = apply_rotary_emb(query_lr, cos_lr, sin_lr)
            key_context = apply_rotary_emb(key_context, cos_context, sin_context)

            attn_output_lr = dispatch_attention_fn(
                query_lr,
                key_context,
                value_context,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            attn_outputs.append(attn_output_lr)

        if has_hr_tokens:
            if rotary_emb_hr is None:
                raise ValueError("Expected rotary_emb_hr when HR tokens are present.")

            cos_hr_full, sin_hr_full = rotary_emb_hr
            if cos_hr_full.shape[1] < seq_len:
                raise ValueError("rotary_emb_hr has insufficient sequence length.")

            cos_hr_tokens = cos_hr_full[:, n_lr:, ...]
            sin_hr_tokens = sin_hr_full[:, n_lr:, ...]

            query_hr = query[:, n_lr:, ...]
            key_hr_tokens = key[:, n_lr:, ...]
            value_hr_tokens = value[:, n_lr:, ...]

            key_hr_context = key_hr_tokens
            value_hr_context = value_hr_tokens
            cos_hr_context = cos_hr_tokens
            sin_hr_context = sin_hr_tokens

            if n_lr > 0:
                key_lr_filtered = key[:, :n_lr, ...] 
                value_lr_filtered = value[:, :n_lr, ...] 
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

            query_hr = apply_rotary_emb(query_hr, cos_hr_tokens, sin_hr_tokens)
            key_hr_context = apply_rotary_emb(key_hr_context, cos_hr_context, sin_hr_context)

            attn_output_hr = dispatch_attention_fn(
                query_hr,
                key_hr_context,
                value_hr_context,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            attn_outputs.append(attn_output_hr)

        if not attn_outputs:
            hidden_states = torch.zeros(
                batch_size, 0, attn.heads, query.shape[-1], dtype=query.dtype, device=query.device
            )
        elif len(attn_outputs) == 1:
            hidden_states = attn_outputs[0]
        else:
            hidden_states = torch.cat(attn_outputs, dim=1)

        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = (
            "The WanAttnProcessor2_0 class is deprecated and will be removed in a future version. "
            "Please use WanAttnProcessor instead. "
        )
        deprecate("WanAttnProcessor2_0", "1.0.0", deprecation_message, standard_warn=False)
        return WanAttnProcessor(*args, **kwargs)


class WanAttention(BaseWanAttention):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.linear_factor = linear_factor
        self.ntk_factor = ntk_factor

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        base_freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            _is_t = len(base_freqs) == 0
            base_freq = get_1d_rotary_pos_embed(
                dim,
                theta=theta,
                linear_factor=1 if _is_t else linear_factor,
                ntk_factor=1 if _is_t else ntk_factor,
                freqs_dtype=freqs_dtype,
            )
            base_freqs.append(base_freq)

        self.register_buffer("base_freqs_t", base_freqs[0], persistent=False)
        self.register_buffer("base_freqs_h", base_freqs[1], persistent=False)
        self.register_buffer("base_freqs_w", base_freqs[2], persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        indices_t = torch.arange(ppf, device=hidden_states.device, dtype=self.base_freqs_t.dtype)
        indices_h = torch.arange(pph, device=hidden_states.device, dtype=self.base_freqs_h.dtype)
        indices_w = torch.arange(ppw, device=hidden_states.device, dtype=self.base_freqs_w.dtype)

        freqs_cos_t, freqs_sin_t = self._compute_axis_embeddings(indices_t, self.base_freqs_t)
        freqs_cos_h, freqs_sin_h = self._compute_axis_embeddings(indices_h, self.base_freqs_h)
        freqs_cos_w, freqs_sin_w = self._compute_axis_embeddings(indices_w, self.base_freqs_w)

        freqs_cos_f = freqs_cos_t.view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos_h.view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos_w.view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin_t.view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin_h.view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin_w.view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin

    def get_rotary_for_coords(
        self,
        t_indices: torch.Tensor,
        h_indices: torch.Tensor,
        w_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns RoPE (cos, sin) embeddings for an arbitrary list of token coordinates.

        Args:
            t_indices: Tensor of shape [N] containing temporal indices in patch space (integer or float).
            h_indices: Tensor of shape [N] containing height indices in patch space (integer or float).
            w_indices: Tensor of shape [N] containing width indices in patch space (integer or float).

        Returns:
            Tuple of tensors (cos, sin), each of shape [1, N, 1, attention_head_dim].
        """
        cos_t, sin_t = self._compute_axis_embeddings(t_indices, self.base_freqs_t)
        cos_h, sin_h = self._compute_axis_embeddings(h_indices, self.base_freqs_h)
        cos_w, sin_w = self._compute_axis_embeddings(w_indices, self.base_freqs_w)

        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1).unsqueeze(0).unsqueeze(2)
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1).unsqueeze(0).unsqueeze(2)

        return cos, sin

    def _compute_axis_embeddings(
        self, positions: torch.Tensor, base_freqs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine and sine rotary factors for a single axis."""
        assert positions.dim() == 1
        # if positions.numel() == 0 or base_freqs.numel() == 0:
        #     zeros = base_freqs.new_zeros((positions.shape[0], base_freqs.shape[0] * 2))
        #     zeros = zeros.float()
        #     return zeros, zeros

        # positions = positions.to(device=base_freqs.device, dtype=base_freqs.dtype)
        positions = positions.to(dtype=base_freqs.dtype)
        base_freqs = base_freqs.to(device=positions.device)
        freqs = torch.outer(positions, base_freqs) # [S, D/2]
        cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() # [S, D]
        sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() # [S, D]
        return cos, sin


@maybe_allow_in_graph
class WanTransformerBlock(BaseWanTransformerBlock):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim)
        self.attn1.set_processor(WanAttnProcessor())
        self.attn2.set_processor(WanAttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        token_info: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if token_info is None:
            return super().forward(hidden_states, encoder_hidden_states, temb, rotary_emb)

        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)

        if token_info is not None:
            attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb, token_info=token_info)
        else:
            attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)

        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states

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
    # assert K != 0

    # Compute the selected column indices for each batch: (B, K)
    arange_n = torch.arange(N, device=mask.device)
    idx = arange_n.unsqueeze(0).expand(B, N)[mask].reshape(B, K)  # (B, K)
    return idx

class WanTransformer3DModel(
    BaseWanTransformer3DModel
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        lr_rope_downsample_hr_tokens (`bool`, defaults to `True`):
            When using mixed-resolution denoising, optionally average each 2x2 group of HR tokens when running the
            LR-position rotary pass so that LR queries attend over coarsened HR context aligned to LR coordinates.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        seperate_attention: bool = True,
        lr_rope_downsample_hr_tokens: bool = True,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
        base_use_rope_lr: bool = False,
    ) -> None:
        inner_dim = num_attention_heads * attention_head_dim
        BaseWanTransformer3DModel.__init__.__wrapped__(
            self,
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        self.seperate_attention = seperate_attention
        self.lr_rope_downsample_hr_tokens = lr_rope_downsample_hr_tokens
        self.base_use_rope_lr = base_use_rope_lr
        if lr_rope_downsample_hr_tokens:
            assert seperate_attention
        if base_use_rope_lr:
            assert not seperate_attention

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.rope2 = WanRotaryPosEmbed(
            attention_head_dim,
            patch_size,
            rope_max_seq_len,
            linear_factor=linear_factor,
            ntk_factor=ntk_factor,
        )
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self._mixed_mask_cache: Optional[Dict[str, Any]] = None

    def _prepare_mixed_mask_state(
        self,
        mask_lr_tokens: torch.Tensor,
        mask_lr_keep_tokens: torch.Tensor,
        ft: int,
        ht: int,
        wt: int,
    ) -> Dict[str, Any]:
        device = mask_lr_tokens.device
        batch_size = mask_lr_tokens.shape[0]

        mask_lr_keep_overlap = mask_lr_keep_tokens & mask_lr_tokens

        keep_lr = mask_lr_keep_tokens.reshape(batch_size, -1)

        f_idx = torch.arange(ft, device=device)
        y_idx = torch.arange(ht, device=device)
        x_idx = torch.arange(wt, device=device)
        Fg, Yg, Xg = torch.meshgrid(f_idx, y_idx, x_idx, indexing="ij")
        Fg_flat = Fg.reshape(-1)
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
        scatter_lr_coords: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        scatter_hr_coords: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        coords_t_lr_all: List[torch.Tensor] = []
        coords_h_lr_all: List[torch.Tensor] = []
        coords_w_lr_all: List[torch.Tensor] = []
        coords_t_hr_all: List[torch.Tensor] = []
        coords_h_hr_all: List[torch.Tensor] = []
        coords_w_hr_all: List[torch.Tensor] = []

        for bidx in range(batch_size):
            keep_lr_b = keep_lr[bidx]
            lr_idx = torch.nonzero(keep_lr_b, as_tuple=False).squeeze(1)
            lr_linear_indices_all.append(lr_idx)

            lr_overlap_flat = mask_lr_keep_overlap[bidx].reshape(-1)
            lr_overlap_flags = lr_overlap_flat.index_select(0, lr_idx).to(torch.bool)
            lr_overlap_token_flags.append(lr_overlap_flags)

            f_b = Fg_flat.index_select(0, lr_idx)
            y_b = Yg_flat.index_select(0, lr_idx)
            x_b = Xg_flat.index_select(0, lr_idx)
            scatter_lr_coords.append((f_b, y_b, x_b))

            coords_t_lr = [f_b]
            coords_h_lr = [y_b]
            coords_w_lr = [x_b]

            coords_t_hr = [f_b]
            coords_h_hr = [y_b * 2]
            coords_w_hr = [x_b * 2]

            sel_hr_b = mask_lr_tokens[bidx]
            f_sel, y_sel, x_sel = torch.nonzero(sel_hr_b, as_tuple=True)
            f_sel_numel = f_sel.numel()
            if f_sel_numel > 0:
                overlap_sel = mask_lr_keep_overlap[bidx][f_sel, y_sel, x_sel]
                repeat_count = f_sel_numel

                f_hr_rep = f_sel.repeat_interleave(4)
                y_hr_base = (2 * y_sel).repeat_interleave(4)
                x_hr_base = (2 * x_sel).repeat_interleave(4)
                y_hr_rep = y_hr_base + dy.repeat(repeat_count)
                x_hr_rep = x_hr_base + dx.repeat(repeat_count)

                hr_overlap_flags = overlap_sel.repeat_interleave(4).to(torch.bool)
                hr_overlap_token_flags.append(hr_overlap_flags)

                lin_hr = f_hr_rep * (Ht2 * Wt2) + y_hr_rep * Wt2 + x_hr_rep
                hr_linear_indices_all.append(lin_hr)

                coords_t_lr.append(f_hr_rep)
                coords_h_lr.append(y_sel.repeat_interleave(4))
                coords_w_lr.append(x_sel.repeat_interleave(4))

                coords_t_hr.append(f_hr_rep)
                coords_h_hr.append(y_hr_rep)
                coords_w_hr.append(x_hr_rep)

                scatter_hr_coords.append((f_hr_rep, y_hr_rep, x_hr_rep))
            else:
                empty_long = torch.empty(0, dtype=torch.long, device=device)
                hr_linear_indices_all.append(empty_long)
                hr_overlap_token_flags.append(torch.empty(0, dtype=torch.bool, device=device))
                scatter_hr_coords.append((empty_long, empty_long, empty_long))

            coords_t_lr_all.append(torch.cat(coords_t_lr, dim=0))
            coords_h_lr_all.append(torch.cat(coords_h_lr, dim=0))
            coords_w_lr_all.append(torch.cat(coords_w_lr, dim=0))

            coords_t_hr_all.append(torch.cat(coords_t_hr, dim=0))
            coords_h_hr_all.append(torch.cat(coords_h_hr, dim=0))
            coords_w_hr_all.append(torch.cat(coords_w_hr, dim=0))

        seq_lens = [coords.numel() for coords in coords_t_lr_all]
        if len(set(seq_lens)) != 1:
            raise ValueError(f"forward_mixed currently requires equal token counts across batch. Got {seq_lens}.")
        N = seq_lens[0]

        lr_lengths = [idx.numel() for idx in lr_linear_indices_all]
        if len(set(lr_lengths)) != 1:
            raise ValueError("forward_mixed expects the same num LR tokens per batch element.")
        n_lr = lr_lengths[0]

        lr_overlap_mask_tensor = torch.stack(lr_overlap_token_flags, dim=0)
        hr_overlap_mask_tensor = torch.stack(hr_overlap_token_flags, dim=0)

        cos_lr_list = []
        sin_lr_list = []
        cos_hr_list = []
        sin_hr_list = []
        for bidx in range(batch_size):
            cos_lr_b, sin_lr_b = self.rope2.get_rotary_for_coords(
                coords_t_lr_all[bidx],
                coords_h_lr_all[bidx],
                coords_w_lr_all[bidx],
            )
            cos_lr_list.append(cos_lr_b.squeeze(0))
            sin_lr_list.append(sin_lr_b.squeeze(0))

            cos_hr_b, sin_hr_b = self.rope2.get_rotary_for_coords(
                coords_t_hr_all[bidx],
                coords_h_hr_all[bidx],
                coords_w_hr_all[bidx],
            )
            cos_hr_list.append(cos_hr_b.squeeze(0))
            sin_hr_list.append(sin_hr_b.squeeze(0))

        rotary_emb_lr = (torch.stack(cos_lr_list, dim=0), torch.stack(sin_lr_list, dim=0))
        rotary_emb_hr_query = (torch.stack(cos_hr_list, dim=0), torch.stack(sin_hr_list, dim=0))

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
            "scatter_lr_coords": tuple(scatter_lr_coords),
            "scatter_hr_coords": tuple(scatter_hr_coords),
            "n_lr": n_lr,
            "N": N,
        }

    def _get_mixed_mask_state(
        self,
        mask_lr_tokens: torch.Tensor,
        mask_lr_keep_tokens: Optional[torch.Tensor],
        batch_size: int,
        ft: int,
        ht: int,
        wt: int,
    ) -> Dict[str, Any]:
        expected_shape = (batch_size, ft, ht, wt)
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
        mask_state = self._prepare_mixed_mask_state(mask_clone, keep_mask, ft, ht, wt)
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
        mask_lr_tokens: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        mask_lr_keep_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mixed-resolution forward path.

        - Uses `mask_lr_tokens` (shape [B, Ft, Ht, Wt] in patch space) to pick which HR sub-tokens (2x2 per selected
          LR token) to include from the HR latent. When `mask_lr_keep_tokens` is provided, it describes which LR tokens
          are kept for the LR branch, allowing overlap with the HR area.
        - Returns two noise predictions: for LR and HR latents respectively, both in 5D grids matching the inputs.

        Notes:
            - Only spatial upscaling by 2x is supported (patch_size expected (1, 2, 2)).
            - Temporal resolution is kept identical in LR and HR latents.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # Validate patch size: we currently rely on (1,2,2)
        p_t, p_h, p_w = self.config.patch_size
        if not (p_t == 1 and p_h == 2 and p_w == 2):
            raise ValueError(
                f"forward_mixed currently supports patch_size=(1,2,2), got {self.config.patch_size}"
            )

        # Shapes
        b_lr, c_lr, f_lr, h_lr, w_lr = hidden_states_lr.shape # [B, 16, 13, 60, 60]
        b_hr, c_hr, f_hr, h_hr, w_hr = hidden_states_hr.shape # [B, 16, 13, 120, 120]
        if (b_lr != b_hr) or (c_lr != c_hr) or (f_lr != f_hr) or (h_hr != 2 * h_lr) or (w_hr != 2 * w_lr):
            raise ValueError(
                "Expected HR latent to have 2x spatial resolution and same batch/channels/frames as LR latent."
            )

        # Compute patch-embedded tokens for LR and HR
        tokens_lr = self.patch_embedding(hidden_states_lr)  # [B, D, Ft, Ht, Wt]   = [B, D, 13, 30, 30]
        tokens_hr = self.patch_embedding(hidden_states_hr)  # [B, D, Ft, 2Ht, 2Wt] = [B, D, 13, 60, 60]

        b, d, ft, ht, wt = tokens_lr.shape
        _, _, ft_hr, ht2, wt2 = tokens_hr.shape
        if ft_hr != ft or ht2 != 2 * ht or wt2 != 2 * wt:
            raise RuntimeError("Unexpected shapes after patch embedding for LR/HR.")

        mask_state = self._get_mixed_mask_state(mask_lr_tokens, mask_lr_keep_tokens, b, ft, ht, wt)

        tokens_lr_flat = tokens_lr.flatten(2).transpose(1, 2)
        tokens_hr_flat = tokens_hr.flatten(2).transpose(1, 2)

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

        mixed_tokens = torch.stack(mixed_tokens_list, dim=0)

        rotary_emb_lr = mask_state["rotary_emb_lr"]
        rotary_emb_hr_query = mask_state["rotary_emb_hr_query"]
        lr_overlap_idx_rev = mask_state["lr_overlap_idx_rev"]
        hr_overlap_idx_rev = mask_state["hr_overlap_idx_rev"]
        has_overlap_hr = mask_state["has_overlap_hr"]
        has_overlap_lr = mask_state["has_overlap_lr"]
        scatter_lr_coords = mask_state["scatter_lr_coords"]
        scatter_hr_coords = mask_state["scatter_hr_coords"]

        Ht = ht
        Wt = wt
        Ht2 = ht * 2
        Wt2 = wt * 2

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            assert ts_seq_len == N
            timestep = timestep.flatten() # batch_size * seq_len
        else:
            ts_seq_len = None

        # Condition embeddings
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
        # timestep_proj: [B, N, 6, inner_dim] = [B, N, 6, D]

        # Transformer blocks over mixed tokens
        hidden_states = mixed_tokens  # [B, N, D]
        
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if n_lr == N or not self.seperate_attention:
            token_info = None
            if self.base_use_rope_lr:
                img_rotary_emb = rotary_emb_lr
            else:
                img_rotary_emb = rotary_emb_hr_query
        else:
            token_info = {
                "n_lr": n_lr,
                "rotary_query": rotary_emb_hr_query,
                "lr_overlap_idx_rev": lr_overlap_idx_rev,
                "hr_overlap_idx_rev": hr_overlap_idx_rev,
                "has_overlap_lr": has_overlap_lr,
                "has_overlap_hr": has_overlap_hr,
                "lr_rope_downsample_hr_tokens": getattr(self.config, "lr_rope_downsample_hr_tokens", False),
            }
            img_rotary_emb = rotary_emb_lr

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                img_rotary_emb,
                token_info=token_info,
            )

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        # hidden_states: [B, N, D]
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        token_patches = self.proj_out(hidden_states)  # [B, N, p_t * p_h * p_w * oc]

        oc = self.config.out_channels or self.config.in_channels
        # TODO: avoid permutation
        token_patches = token_patches.view(b, N, p_t*p_h*p_w, -1).permute(0, 1, 3, 2)
        token_patches = token_patches.view(b, N, oc, p_t, p_h, p_w).contiguous() # [B, N, 16, 1, 2, 2]

        # Now, scatter tokens back into LR and HR patch grids (token space)
        # Initialize zeros grids in patch space tokens: [B, Ft, Ht, Wt, oc, p_t, p_h, p_w]
        pred_lr_tokens = torch.zeros(
            b, ft, ht, wt, oc, p_t, p_h, p_w, device=hidden_states_lr.device, dtype=hidden_states_lr.dtype
        ) # [B, 13, 30, 30, 16, 1, 1, 2]
        pred_hr_tokens = torch.zeros(
            b, ft, Ht2, Wt2, oc, p_t, p_h, p_w, device=hidden_states_lr.device, dtype=hidden_states_lr.dtype
        ) # [B, 13, 60, 60, 16, 1, 1, 2]

        # Split first segment (LR tokens) and second segment (HR tokens) per batch

        # LR placement
        lr_patches = token_patches[:, :n_lr]  # [B, n_lr, oc, p_t, p_h, p_w]
        for bidx in range(b):
            f_b, y_b, x_b = scatter_lr_coords[bidx]
            pred_lr_tokens[bidx, f_b, y_b, x_b, :, :, :, :] = lr_patches[bidx]

        # HR placement
        hr_patches = token_patches[:, n_lr:]  # [B, n_hr, oc, p_t, p_h, p_w]
        for bidx in range(b):
            f2, y2, x2 = scatter_hr_coords[bidx]
            # if f2.numel() == 0:
            #     continue
            pred_hr_tokens[bidx, f2, y2, x2, :, :, :, :] = hr_patches[bidx]

        # Unpatchify: reshape patch grids back to latent shapes
        # LR token grid -> [B, oc, Ft, p_t, Ht, p_h, Wt, p_w] -> [B, oc, F, H, W]
        pred_lr = pred_lr_tokens.permute(0, 4, 1, 5, 2, 6, 3, 7)
        pred_lr = pred_lr.flatten(2, 3).flatten(3, 4).flatten(4, 5) # [B, 16, 13, 60, 60]

        # HR token grid -> [B, oc, Ft, p_t, 2Ht, p_h, 2Wt, p_w] -> [B, oc, F, 2H, 2W]
        pred_hr = pred_hr_tokens.permute(0, 4, 1, 5, 2, 6, 3, 7)
        pred_hr = pred_hr.flatten(2, 3).flatten(3, 4).flatten(4, 5) # [B, 16, 13, 120, 120]

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        return pred_lr, pred_hr
