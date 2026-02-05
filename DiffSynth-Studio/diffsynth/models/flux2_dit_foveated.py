"""
FLUX2 DiT with Cross-Resolution Phase-Aligned (CRPA) Attention for Foveated Generation.

This module integrates foveation logic from transformer_flux2_foveation.py into the DiffSynth FLUX2 architecture.
Key features:
- Phase-aligned RoPE embeddings for mixed-resolution latents
- CRPA attention mechanism that handles high-res and low-res tokens differently
- Support for both double-stream and single-stream transformer blocks
"""

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..core.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward

# Import base classes and utilities from flux2_dit
from .flux2_dit import (
    get_timestep_embedding,
    TimestepEmbedding,
    Timesteps,
    AdaLayerNormContinuous,
    get_1d_rotary_pos_embed,
    apply_rotary_emb,
    _get_projections,
    _get_qkv_projections,
    Flux2SwiGLU,
    Flux2FeedForward,
    Flux2PosEmbed,
    Flux2TimestepGuidanceEmbeddings,
    Flux2Modulation,
)


# =====================================================================
# CRPA (Cross-Resolution Phase-Aligned) Attention Processor for FLUX2
# =====================================================================

class Flux2CRPAAttnProcessor:
    """
    Cross-Resolution Phase-Aligned Attention Processor for FLUX2.
    Implements 'One Attention, One Scale' for foveated generation.
    
    This processor handles mixed-resolution latents by:
    1. Using high-res RoPE for high-resolution query tokens attending to all keys
    2. Using low-res RoPE for low-resolution query tokens attending to subsampled keys
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0.")

    def __call__(
        self,
        attn: "Flux2CRPAAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[Any, Any, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            attn: The attention module
            hidden_states: Image tokens [B, img_seq_len, C]
            encoder_hidden_states: Text tokens [B, txt_seq_len, C]
            attention_mask: Optional attention mask
            image_rotary_emb: Tuple of (HR_RoPE, LR_RoPE, resolution_mask, resolution_mask_top_left)
                - HR_RoPE: (cos_hr, sin_hr) for high-resolution grid
                - LR_RoPE: (cos_lr, sin_lr) for low-resolution grid  
                - resolution_mask: Boolean mask (True=HR, False=LR) for each token
                - resolution_mask_top_left: Boolean mask for key subsampling
        """
        # 1. Projections
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Handle encoder (Text) concatenation for double-stream blocks
        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # 2. Phase-Aligned RoPE
        if image_rotary_emb is not None:
            rope_hr, rope_lr, res_mask, res_mask_top_left = image_rotary_emb
            
            if res_mask is not None:
                # Phase-aligned attention for mixed-resolution
                if res_mask.shape[0] != query.shape[1]:
                    raise ValueError(f"Resolution mask shape {res_mask.shape} mismatch with sequence {query.shape[1]}")

                is_hr = (res_mask > 0.5)  # Boolean mask for HR spatial regions + text tokens
                is_lr = ~is_hr

                # HR query path: Q_HR attends to K using HR grid phases
                if is_hr.any():
                    rope_hr_q = (rope_hr[0][is_hr], rope_hr[1][is_hr])
                    q_hr = apply_rotary_emb(query[:, is_hr, ...], rope_hr_q, sequence_dim=1)
                    k_hr = apply_rotary_emb(key, rope_hr, sequence_dim=1)
                    
                    out_hr = attention_forward(
                        q_hr, k_hr, value,
                        q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
                    )
                else:
                    out_hr = None

                # LR query path: Q_LR attends to subsampled K using LR grid phases
                if is_lr.any():
                    rope_lr_q = (rope_lr[0][is_lr], rope_lr[1][is_lr])
                    q_lr = apply_rotary_emb(query[:, is_lr, ...], rope_lr_q, sequence_dim=1)
                    
                    # Subsample LR and HR top-left tokens for key/value
                    key_sampled = key[:, res_mask_top_left, ...]
                    value_sampled = value[:, res_mask_top_left, ...]
                    rope_lr_k = (rope_lr[0][res_mask_top_left], rope_lr[1][res_mask_top_left])
                    k_lr = apply_rotary_emb(key_sampled, rope_lr_k, sequence_dim=1)

                    out_lr = attention_forward(
                        q_lr, k_lr, value_sampled,
                        q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
                    )
                else:
                    out_lr = None

                # Reassemble hidden states
                hidden_states = torch.zeros_like(query)
                if out_hr is not None:
                    hidden_states[:, is_hr, ...] = out_hr.to(hidden_states.dtype)
                if out_lr is not None:
                    hidden_states[:, is_lr, ...] = out_lr.to(hidden_states.dtype)
            else:
                # No resolution mask provided, use full high-res RoPE
                query = apply_rotary_emb(query, rope_hr, sequence_dim=1)
                key = apply_rotary_emb(key, rope_hr, sequence_dim=1)
                hidden_states = attention_forward(
                    query, key, value,
                    q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
                )
        else:
            # No RoPE at all
            hidden_states = attention_forward(
                query, key, value,
                q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
            )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class Flux2CRPAParallelSelfAttnProcessor:
    """
    Cross-Resolution Phase-Aligned Attention Processor for FLUX2 single-stream blocks.
    Handles the parallel self-attention + MLP architecture of single-stream blocks.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0.")

    def __call__(
        self,
        attn: "Flux2CRPAParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Parallel in (QKV + MLP in) projection
        hidden_states_proj = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states_proj, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        # Handle the attention logic
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Phase-Aligned RoPE for single-stream blocks
        if image_rotary_emb is not None:
            rope_hr, rope_lr, res_mask, res_mask_top_left = image_rotary_emb
            
            if res_mask is not None:
                if res_mask.shape[0] != query.shape[1]:
                    raise ValueError(f"Resolution mask shape {res_mask.shape} mismatch with sequence {query.shape[1]}")

                is_hr = (res_mask > 0.5)
                is_lr = ~is_hr

                # HR query path
                if is_hr.any():
                    rope_hr_q = (rope_hr[0][is_hr], rope_hr[1][is_hr])
                    q_hr = apply_rotary_emb(query[:, is_hr, ...], rope_hr_q, sequence_dim=1)
                    k_hr = apply_rotary_emb(key, rope_hr, sequence_dim=1)
                    
                    out_hr = attention_forward(
                        q_hr, k_hr, value,
                        q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
                    )
                else:
                    out_hr = None

                # LR query path
                if is_lr.any():
                    rope_lr_q = (rope_lr[0][is_lr], rope_lr[1][is_lr])
                    q_lr = apply_rotary_emb(query[:, is_lr, ...], rope_lr_q, sequence_dim=1)
                    
                    key_sampled = key[:, res_mask_top_left, ...]
                    value_sampled = value[:, res_mask_top_left, ...]
                    rope_lr_k = (rope_lr[0][res_mask_top_left], rope_lr[1][res_mask_top_left])
                    k_lr = apply_rotary_emb(key_sampled, rope_lr_k, sequence_dim=1)

                    out_lr = attention_forward(
                        q_lr, k_lr, value_sampled,
                        q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
                    )
                else:
                    out_lr = None

                # Reassemble
                attn_hidden_states = torch.zeros_like(query)
                if out_hr is not None:
                    attn_hidden_states[:, is_hr, ...] = out_hr.to(attn_hidden_states.dtype)
                if out_lr is not None:
                    attn_hidden_states[:, is_lr, ...] = out_lr.to(attn_hidden_states.dtype)
                hidden_states = attn_hidden_states
            else:
                # No resolution mask, use standard RoPE
                query = apply_rotary_emb(query, rope_hr, sequence_dim=1)
                key = apply_rotary_emb(key, rope_hr, sequence_dim=1)
                hidden_states = attention_forward(
                    query, key, value,
                    q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
                )
        else:
            hidden_states = attention_forward(
                query, key, value,
                q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
            )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Handle the feedforward (FF) logic
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        # Concatenate and parallel output projection
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states


# =====================================================================
# Attention Modules with CRPA Support
# =====================================================================

class Flux2CRPAAttention(torch.nn.Module):
    """FLUX2 Attention with support for CRPA processor."""
    _default_processor_cls = Flux2CRPAAttnProcessor

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.use_bias = bias
        self.dropout = dropout

        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        # QK Norm
        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


class Flux2CRPAParallelSelfAttention(torch.nn.Module):
    """FLUX2 parallel self-attention for single-stream blocks with CRPA support."""
    _default_processor_cls = Flux2CRPAParallelSelfAttnProcessor

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        mlp_ratio: float = 4.0,
        mlp_mult_factor: int = 2,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.use_bias = bias
        self.dropout = dropout

        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(query_dim * self.mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor

        # Fused QKV projections + MLP input projection
        self.to_qkv_mlp_proj = torch.nn.Linear(
            self.query_dim, self.inner_dim * 3 + self.mlp_hidden_dim * self.mlp_mult_factor, bias=bias
        )
        self.mlp_act_fn = Flux2SwiGLU()

        # QK Norm
        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        # Fused attention output projection + MLP output projection
        self.to_out = torch.nn.Linear(self.inner_dim + self.mlp_hidden_dim, self.out_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb, **kwargs)


# =====================================================================
# Transformer Blocks with CRPA Support
# =====================================================================

class Flux2CRPASingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = Flux2CRPAParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
            processor=Flux2CRPAParallelSelfAttnProcessor(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb_mod_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        split_hidden_states: bool = False,
        text_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        mod_shift, mod_scale, mod_gate = temb_mod_params

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if split_hidden_states:
            encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
            return encoder_hidden_states, hidden_states
        else:
            return hidden_states


class Flux2CRPATransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = Flux2CRPAAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
            processor=Flux2CRPAAttnProcessor(),
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_params_img: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        temb_mod_params_txt: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}

        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

        # Img stream
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        # Conditioning txt stream
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        # Attention on concatenated img + txt stream
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        attn_output, context_attn_output = attention_outputs

        # Process attention outputs for the image stream
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        # Process attention outputs for the text stream
        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


# =====================================================================
# Main FLUX2 DiT with Foveation Support
# =====================================================================

class Flux2DiTFoveated(torch.nn.Module):
    """
    FLUX2 DiT with Phase-Aligned Attention for Foveated Generation.
    
    This model extends the standard FLUX2 DiT with support for:
    - Mixed-resolution latent processing
    - Phase-aligned RoPE for consistent position encoding across resolutions
    - CRPA attention mechanism for efficient foveated rendering
    """

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # 1. Sinusoidal positional embedding for RoPE
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)

        # 2. Combined timestep + guidance embedding
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=guidance_embeds,
        )

        # 3. Modulation layers
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        # 4. Input projections
        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        # 5. Double Stream Transformer Blocks (with CRPA)
        self.transformer_blocks = nn.ModuleList(
            [
                Flux2CRPATransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Single Stream Transformer Blocks (with CRPA)
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2CRPASingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 7. Output layers
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        # Foveation parameters
        resolution_mask: Optional[torch.Tensor] = None,
        resolution_mask_top_left: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with support for foveated generation via resolution_mask.
        
        Args:
            hidden_states: Image latent tokens [B, img_seq_len, C]
            encoder_hidden_states: Text tokens [B, txt_seq_len, C]
            timestep: Denoising timestep
            img_ids: Image position IDs [B, img_seq_len, 4] (T, H, W, L format)
            txt_ids: Text position IDs [B, txt_seq_len, 4]
            guidance: Guidance scale
            joint_attention_kwargs: Dict containing optional kwargs
            resolution_mask: [img_seq_len] tensor (1=HR, 0=LR)
            resolution_mask_top_left: [num_tokens] tensor for key subsampling
        """
        # 0. Handle input arguments
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        num_txt_tokens = encoder_hidden_states.shape[1]

        # 1. Calculate timestep embedding and modulation parameters
        timestep = timestep.to(hidden_states.dtype) * 1000

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)[0]

        # 2. Input projection
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 3. Calculate RoPE embeddings with phase alignment for foveation
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        # Create HR and LR position IDs
        # For FLUX2: img_ids has shape [S, 4] with (T, H, W, L) format
        ids_hr = torch.cat((txt_ids, img_ids), dim=0)
        
        # For LR grid, divide H and W coordinates by 2 (don't touch T and L)
        img_ids_lr = img_ids.clone()
        img_ids_lr[:, 1] = img_ids_lr[:, 1] / 2.0  # H coordinate
        img_ids_lr[:, 2] = img_ids_lr[:, 2] / 2.0  # W coordinate
        ids_lr = torch.cat((txt_ids, img_ids_lr), dim=0)

        # Compute dual RoPE embeddings
        rope_cos_hr, rope_sin_hr = self.pos_embed(ids_hr)
        rope_cos_lr, rope_sin_lr = self.pos_embed(ids_lr)

        # Pack into structure for CRPA Processor
        if resolution_mask is not None:
            if resolution_mask.ndim == 2:
                resolution_mask = resolution_mask[0]
            # Create text mask (all ones - text is always "high-res")
            txt_mask = torch.ones(txt_ids.shape[0], device=resolution_mask.device, dtype=resolution_mask.dtype)
            full_res_mask = torch.cat((txt_mask, resolution_mask), dim=0).bool()
            full_res_mask_top_left = torch.cat((txt_mask, resolution_mask_top_left), dim=0).bool()
            image_rotary_emb = ((rope_cos_hr, rope_sin_hr), (rope_cos_lr, rope_sin_lr), full_res_mask, full_res_mask_top_left)
        else:
            # Standard path without foveation
            image_rotary_emb = ((rope_cos_hr, rope_sin_hr), None, None, None)
        
        # For single-stream blocks, use same CRPA structure
        concat_rotary_emb = image_rotary_emb

        # 4. Double Stream Transformer Blocks
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        
        # Concatenate text and image streams for single-block inference
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 5. Single Stream Transformer Blocks
        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        
        # Remove text tokens from concatenated stream
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        # 6. Output layers
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return output
