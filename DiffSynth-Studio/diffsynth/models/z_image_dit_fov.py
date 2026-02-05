"""
Z-Image DiT with Foveated Attention (Cross-Resolution Phase-Aligned Attention)

This module implements the foveated diffusion transformer that supports mixed-resolution
tokens with phase-aligned RoPE embeddings. Based on the CRPA (Cross-Resolution Phase-Aligned)
attention mechanism from the FLUX foveation implementation.

Key features:
- Dual RoPE embeddings for HR (high-resolution) and LR (low-resolution) tokens
- Phase-aligned attention where:
  - HR queries attend to all keys using HR RoPE
  - LR queries attend to subsampled keys (HR top-left + LR) using LR RoPE
- Compatible with Z-Image's coordinate system and sequence structure
"""

import math
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from torch.nn.utils.rnn import pad_sequence
from torchvision.utils import save_image

from .general_modules import RMSNorm
from ..core.attention import attention_forward
from ..core.device.npu_compatible_device import IS_NPU_AVAILABLE, get_device_type
from ..core.gradient import gradient_checkpoint_forward



ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32
X_PAD_DIM = 64


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast(get_device_type(), enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(torch.bfloat16))
        return t_emb


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return torch_F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class RopeEmbedderFoveated:
    """
    RoPE embedder that supports foveated (mixed-resolution) position embeddings.
    
    For foveation, we compute two sets of RoPE:
    - HR (high-resolution): Standard position embeddings
    - LR (low-resolution): Position embeddings with scaled-down coordinates (/ 2)
    """
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
                freqs_cis.append(freqs_cis_i)
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        """Compute RoPE embeddings for given position IDs."""
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i].long()
            if IS_NPU_AVAILABLE:
                result.append(torch.index_select(self.freqs_cis[i], 0, index))
            else:
                result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)

    def forward_dual(self, ids_hr: torch.Tensor, ids_lr: torch.Tensor):
        """
        Compute dual RoPE embeddings for HR and LR position IDs.
        
        Args:
            ids_hr: High-resolution position IDs [seq_len, 3]
            ids_lr: Low-resolution position IDs [seq_len, 3] (typically ids_hr with spatial dims / 2)
            
        Returns:
            freqs_cis_hr: HR RoPE embeddings
            freqs_cis_lr: LR RoPE embeddings
        """
        freqs_cis_hr = self(ids_hr)
        freqs_cis_lr = self(ids_lr)
        return freqs_cis_hr, freqs_cis_lr


class CRPAAttention(torch.nn.Module):
    """
    Cross-Resolution Phase-Aligned Attention.
    
    Implements the 'One Attention, One Scale' mechanism where:
    - HR (high-resolution) queries attend to all keys using HR RoPE
    - LR (low-resolution) queries attend to subsampled keys using LR RoPE
    
    This allows mixing high and low resolution tokens in the same sequence
    while maintaining proper positional relationships.
    """

    def __init__(self, q_dim, num_heads, head_dim, kv_dim=None, bias_q=False, bias_kv=False, bias_out=False):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = torch.nn.Linear(q_dim, dim_inner, bias=bias_q)
        self.to_k = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_v = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(dim_inner, q_dim, bias=bias_out)])

        self.norm_q = RMSNorm(head_dim, eps=1e-5)
        self.norm_k = RMSNorm(head_dim, eps=1e-5)

    @staticmethod
    def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings using complex multiplication."""
        with torch.amp.autocast(get_device_type(), enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        # Foveation parameters
        freqs_cis_hr: torch.Tensor = None,
        freqs_cis_lr: torch.Tensor = None,
        resolution_mask: torch.Tensor = None,
        resolution_mask_top_left: torch.Tensor = None,
    ):
        """
        Forward pass with optional CRPA (Cross-Resolution Phase-Aligned) attention.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, dim]
            freqs_cis: Standard RoPE (used when no foveation)
            attention_mask: Attention mask
            freqs_cis_hr: HR RoPE embeddings for foveation
            freqs_cis_lr: LR RoPE embeddings for foveation
            resolution_mask: Boolean mask where True = HR token, False = LR token
            resolution_mask_top_left: Boolean mask for K/V subsampling (HR top-left + LR tokens)
        """
        batch_size = hidden_states.shape[0]
        
        # Project Q, K, V
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.num_heads, -1))
        key = key.unflatten(-1, (self.num_heads, -1))
        value = value.unflatten(-1, (self.num_heads, -1))

        # Apply QK norms
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Check if we should use CRPA
        use_crpa = (
            resolution_mask is not None and 
            freqs_cis_hr is not None and 
            freqs_cis_lr is not None
        )

        if use_crpa:
            # Cross-Resolution Phase-Aligned Attention
            hidden_states = self._forward_crpa(
                query, key, value,
                freqs_cis_hr, freqs_cis_lr,
                resolution_mask, resolution_mask_top_left,
                attention_mask
            )
        else:
            # Standard attention with single RoPE
            if freqs_cis is not None:
                query = self.apply_rotary_emb(query, freqs_cis)
                key = self.apply_rotary_emb(key, freqs_cis)

            dtype = query.dtype
            query, key = query.to(dtype), key.to(dtype)

            hidden_states = attention_forward(
                query, key, value,
                q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
                attn_mask=attention_mask,
            )

        # Reshape and project output
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        output = self.to_out[0](hidden_states)
        if len(self.to_out) > 1:
            output = self.to_out[1](output)

        return output

    def _forward_crpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        freqs_cis_hr: torch.Tensor,
        freqs_cis_lr: torch.Tensor,
        resolution_mask: torch.Tensor,
        resolution_mask_top_left: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """
        Cross-Resolution Phase-Aligned attention forward.
        
        HR tokens: Q uses HR RoPE, attends to all K (with HR RoPE)
        LR tokens: Q uses LR RoPE, attends to subsampled K (HR top-left + LR, with LR RoPE)
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Ensure resolution_mask matches sequence length
        if resolution_mask.shape[0] != seq_len:
            raise ValueError(f"Resolution mask shape {resolution_mask.shape} mismatch with sequence {seq_len}")

        is_hr = resolution_mask.bool()  # True for HR tokens
        is_lr = ~is_hr  # True for LR tokens

        # Initialize output tensor
        hidden_states = torch.zeros_like(query)

        # Helper function to handle both 2D [seq, dim] and 3D [batch, seq, dim] freqs_cis
        def get_rope(freqs_cis, mask):
            """Select RoPE for given mask positions, handling 2D or 3D input."""
            if freqs_cis.ndim == 2:
                # [seq, dim] -> select positions -> [num_selected, dim] -> [batch, num_selected, dim]
                return freqs_cis[mask].unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # [batch, seq, dim] -> select positions -> [batch, num_selected, dim]
                return freqs_cis[:, mask, :]

        def get_rope_full(freqs_cis):
            """Get full RoPE, handling 2D or 3D input."""
            if freqs_cis.ndim == 2:
                # [seq, dim] -> [batch, seq, dim]
                return freqs_cis.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # [batch, seq, dim] already
                return freqs_cis

        # Process HR tokens
        if is_hr.any():
            # HR queries attend to ALL keys using HR RoPE
            rope_hr_q = get_rope(freqs_cis_hr, is_hr)  # [batch, num_hr, dim]
            rope_hr_k = get_rope_full(freqs_cis_hr)  # [batch, seq, dim]
            
            q_hr = self.apply_rotary_emb(query[:, is_hr, ...], rope_hr_q)
            k_hr = self.apply_rotary_emb(key, rope_hr_k)

            out_hr = attention_forward(
                q_hr, k_hr, value,
                q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
                attn_mask=attention_mask,
            )
            hidden_states[:, is_hr, ...] = out_hr.to(hidden_states.dtype)

        # Process LR tokens
        if is_lr.any():
            # LR queries attend to SUBSAMPLED keys (HR top-left + LR) using LR RoPE
            rope_lr_q = get_rope(freqs_cis_lr, is_lr)  # [batch, num_lr, dim]
            
            q_lr = self.apply_rotary_emb(query[:, is_lr, ...], rope_lr_q)

            # Subsample K and V using top-left mask
            key_sampled = key[:, resolution_mask_top_left, ...]
            value_sampled = value[:, resolution_mask_top_left, ...]
            rope_lr_k = get_rope(freqs_cis_lr, resolution_mask_top_left)  # [batch, num_sampled, dim]
            k_lr = self.apply_rotary_emb(key_sampled, rope_lr_k)

            out_lr = attention_forward(
                q_lr, k_lr, value_sampled,
                q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
                attn_mask=None,  # Subsampled, so mask may not apply directly
            )
            hidden_states[:, is_lr, ...] = out_lr.to(hidden_states.dtype)

        return hidden_states


def select_per_token(
    value_noisy: torch.Tensor,
    value_clean: torch.Tensor,
    noise_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    noise_mask_expanded = noise_mask.unsqueeze(-1)
    return torch.where(
        noise_mask_expanded == 1,
        value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
        value_clean.unsqueeze(1).expand(-1, seq_len, -1),
    )


class ZImageFoveatedTransformerBlock(nn.Module):
    """
    Transformer block with support for foveated (CRPA) attention.
    """
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        self.attention = CRPAAttention(
            q_dim=dim,
            num_heads=n_heads,
            head_dim=dim // n_heads,
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        noise_mask: Optional[torch.Tensor] = None,
        adaln_noisy: Optional[torch.Tensor] = None,
        adaln_clean: Optional[torch.Tensor] = None,
        # Foveation parameters
        freqs_cis_hr: torch.Tensor = None,
        freqs_cis_lr: torch.Tensor = None,
        resolution_mask: torch.Tensor = None,
        resolution_mask_top_left: torch.Tensor = None,
    ):
        if self.modulation:
            seq_len = x.shape[1]

            if noise_mask is not None:
                mod_noisy = self.adaLN_modulation(adaln_noisy)
                mod_clean = self.adaLN_modulation(adaln_clean)

                scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
                scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)

                gate_msa_noisy, gate_mlp_noisy = gate_msa_noisy.tanh(), gate_mlp_noisy.tanh()
                gate_msa_clean, gate_mlp_clean = gate_msa_clean.tanh(), gate_mlp_clean.tanh()

                scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
                scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean

                scale_msa = select_per_token(scale_msa_noisy, scale_msa_clean, noise_mask, seq_len)
                scale_mlp = select_per_token(scale_mlp_noisy, scale_mlp_clean, noise_mask, seq_len)
                gate_msa = select_per_token(gate_msa_noisy, gate_msa_clean, noise_mask, seq_len)
                gate_mlp = select_per_token(gate_mlp_noisy, gate_mlp_clean, noise_mask, seq_len)
            else:
                mod = self.adaLN_modulation(adaln_input)
                scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
                gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
                scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block with CRPA support
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
                freqs_cis_hr=freqs_cis_hr,
                freqs_cis_lr=freqs_cis_lr,
                resolution_mask=resolution_mask,
                resolution_mask_top_left=resolution_mask_top_left,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            # Attention block with CRPA support
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
                freqs_cis_hr=freqs_cis_hr,
                freqs_cis_lr=freqs_cis_lr,
                resolution_mask=resolution_mask,
                resolution_mask_top_left=resolution_mask_top_left,
            )
            x = x + self.attention_norm2(attn_out)

            # FFN block
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c=None, noise_mask=None, c_noisy=None, c_clean=None):
        seq_len = x.shape[1]

        if noise_mask is not None:
            scale_noisy = 1.0 + self.adaLN_modulation(c_noisy)
            scale_clean = 1.0 + self.adaLN_modulation(c_clean)
            scale = select_per_token(scale_noisy, scale_clean, noise_mask, seq_len)
        else:
            assert c is not None, "Either c or (c_noisy, c_clean) must be provided"
            scale = 1.0 + self.adaLN_modulation(c)
            scale = scale.unsqueeze(1)

        x = self.norm_final(x) * scale
        x = self.linear(x)
        return x


class ZImageDiTFoveated(nn.Module):
    """
    Z-Image DiT with Foveated (CRPA) Attention support.
    
    This model extends ZImageDiT to support mixed-resolution token sequences
    using Cross-Resolution Phase-Aligned attention.
    """
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageFoveatedTransformerBlock"]

    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        siglip_feat_dim=None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        
        self.noise_refiner = nn.ModuleList([
            ZImageFoveatedTransformerBlock(
                1000 + layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True,
            )
            for layer_id in range(n_refiner_layers)
        ])
        
        self.context_refiner = nn.ModuleList([
            ZImageFoveatedTransformerBlock(
                layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False,
            )
            for layer_id in range(n_refiner_layers)
        ])
        
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        # Optional SigLIP components
        self.siglip_feat_dim = siglip_feat_dim
        if siglip_feat_dim is not None:
            self.siglip_embedder = nn.Sequential(
                RMSNorm(siglip_feat_dim, eps=norm_eps), nn.Linear(siglip_feat_dim, dim, bias=True)
            )
            self.siglip_refiner = nn.ModuleList([
                ZImageFoveatedTransformerBlock(
                    2000 + layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ])
            self.siglip_pad_token = nn.Parameter(torch.empty((1, dim)))
        else:
            self.siglip_embedder = None
            self.siglip_refiner = None
            self.siglip_pad_token = None

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList([
            ZImageFoveatedTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
            for layer_id in range(n_layers)
        ])
        
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedderFoveated(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        """Patchify images and prepare position IDs."""
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            # Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            all_cap_pad_mask.append(
                torch.cat([
                    torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                ], dim=0)
            )
            cap_padded_feat = torch.cat([cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)], dim=0)
            all_cap_feats_out.append(cap_padded_feat)

            # Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            
            all_image_pad_mask.append(
                torch.cat([
                    torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                ], dim=0)
            )
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return all_image_out, all_cap_feats_out, {
            "x_size": all_image_size,
            "x_pos_ids": all_image_pos_ids,
            "cap_pos_ids": all_cap_pos_ids,
            "x_pad_mask": all_image_pad_mask,
            "cap_pad_mask": all_cap_pad_mask
        }

    def patchify_and_embed_foveated(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        foveation_mask: torch.Tensor,
        patch_size: int = 2,
        f_patch_size: int = 1,
        token_downsampler: nn.Module = None,
        timestep: torch.Tensor = None,
    ):
        """
        Patchify images with foveation mask support and spatial-domain downsampling.
        
        Creates mixed-resolution token sequences where:
        - HR regions keep all 4 tokens per 2x2 block
        - LR regions: 4 tokens are unpatchified to 4x4 spatial, downsampled to 2x2,
          then repatchified to 1 token. This ensures the token contains properly
          downsampled content (not just the top-left corner).
        
        Args:
            all_image: List of image tensors [C, F, H, W]
            all_cap_feats: List of caption feature tensors
            foveation_mask: Mask tensor [H_latent, W_latent] where 1=HR, 0=LR
            patch_size: Spatial patch size
            f_patch_size: Temporal patch size
            token_downsampler: Module to downsample 4x4 -> 2x2 in spatial domain
            timestep: Current timestep (for conditioned downsamplers)
            
        Returns:
            Patchified outputs with resolution masks
        """
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids_hr = []
        all_image_pos_ids_lr = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []
        all_resolution_mask = []
        all_resolution_mask_top_left = []
        all_valid_tokens = []  # For expanding back to full res
        all_full_seqlens = []  # Original full sequence lengths
        all_spatial_dims = []  # (F_tokens, H_tokens, W_tokens, h_2, w_2)
        all_is_low_res_block = []  # Boolean mask for LR blocks

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            # Process Caption (same as standard)
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            all_cap_pad_mask.append(
                torch.cat([
                    torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                ], dim=0)
            )
            cap_padded_feat = torch.cat([cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)], dim=0)
            all_cap_feats_out.append(cap_padded_feat)

            # Process Image with foveation
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            # Patchify image
            image_patched = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            image_patched = image_patched.permute(1, 3, 5, 2, 4, 6, 0).reshape(
                F_tokens * H_tokens * W_tokens, pF * pH * pW * C
            )

            # Create position IDs for HR and LR
            h_2, w_2 = H_tokens // 2, W_tokens // 2
            
            # Create full position grid
            pos_ids_full = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)  # [F_t * H_t * W_t, 3]

            # Create HR and LR position IDs
            pos_ids_hr = pos_ids_full.clone()
            pos_ids_lr = pos_ids_full.clone().float()
            pos_ids_lr[..., 1:] = pos_ids_lr[..., 1:] / 2.0  # Scale spatial dims for LR
            pos_ids_lr = pos_ids_lr.to(pos_ids_full.dtype)

            # Process foveation mask into 2x2 blocks
            # foveation_mask is [H_latent, W_latent], we need to map to token space
            mask = foveation_mask.to(device)
            if mask.shape[0] == H_tokens and mask.shape[1] == W_tokens:
                # Mask is already in token space
                mask_blocks = mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
            else:
                # Resize mask to token space
                mask_resized = torch_F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(H_tokens, W_tokens),
                    mode='nearest'
                ).squeeze()
                mask_blocks = mask_resized.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)

            is_high_res_block = (mask_blocks.sum(dim=-1) > 0)  # [h_2, w_2]
            is_low_res_block = ~is_high_res_block

            # Create resolution mask (1 for HR, 0 for LR)
            res_mask = torch.ones(h_2, w_2, 4, device=device)
            res_mask[is_low_res_block, :] = 0.0

            # Create validity mask (which tokens to keep)
            valid_tokens = torch.ones(h_2, w_2, 4, device=device, dtype=torch.bool)
            valid_tokens[is_low_res_block, 1:] = False  # LR blocks only keep index 0

            # Create top-left mask for K/V subsampling
            tl_mask = torch.zeros(h_2, w_2, 4, device=device, dtype=torch.bool)
            tl_mask[..., 0] = True  # Keep top-left for all
            tl_mask[is_low_res_block, :] = True  # Keep all slots marked for LR (but only 0 is valid)

            # Reshape to match image token order [H_t, W_t] = [h_2*2, w_2*2]
            valid_tokens = valid_tokens.view(h_2, w_2, 2, 2).permute(0, 2, 1, 3).reshape(H_tokens * W_tokens)
            res_mask = res_mask.view(h_2, w_2, 2, 2).permute(0, 2, 1, 3).reshape(H_tokens * W_tokens)
            tl_mask = tl_mask.view(h_2, w_2, 2, 2).permute(0, 2, 1, 3).reshape(H_tokens * W_tokens)

            # ================================================================
            # SPATIAL DOMAIN DOWNSAMPLING FOR LR BLOCKS
            # For LR 2x2 token blocks:
            # 1. Unpatchify 4 tokens to 4x4 spatial [4, 4, C]
            # 2. Downsample to 2x2 spatial [2, 2, C] 
            # 3. Repatchify to 1 token [64] (2*2*C)
            # This ensures LR tokens contain properly downsampled content
            # ================================================================
            
            # Reshape image_patched to block format: [h_2, 2, w_2, 2, token_dim]
            token_dim = pF * pH * pW * C  # 64 for typical Z-Image (1*2*2*16)
            image_blocks = image_patched.view(H_tokens, W_tokens, token_dim)
            image_blocks = image_blocks.view(h_2, 2, w_2, 2, token_dim)
            image_blocks = image_blocks.permute(0, 2, 1, 3, 4)  # [h_2, w_2, 2, 2, token_dim]
            
            # For spatial downsampling, we need to unpatchify each 2x2 block to 4x4 spatial
            # Each token [token_dim=64] represents a 2x2 spatial patch of C channels
            # 4 tokens = 4 x (2x2 patch) = 4x4 spatial region
            
            if token_downsampler is not None:
                # Create properly downsampled tokens for LR blocks
                # image_blocks: [h_2, w_2, 2, 2, token_dim]
                # For each LR block, unpatchify 4 tokens to 4x4 spatial, downsample, repatchify
                
                # Get LR block indices
                lr_indices = torch.where(is_low_res_block)  # (lr_h_indices, lr_w_indices)
                
                if len(lr_indices[0]) > 0:
                    # Extract LR blocks: [num_lr, 2, 2, token_dim]
                    lr_blocks_tokens = image_blocks[lr_indices[0], lr_indices[1]]  # [num_lr, 2, 2, token_dim]
                    num_lr = lr_blocks_tokens.shape[0]
                    
                    # Unpatchify each LR block's 4 tokens to 4x4 spatial
                    # Each token: [token_dim] = [pF*pH*pW*C] = [1*2*2*16] = [64]
                    # Represents a 2x2 patch of 16 channels
                    lr_blocks_tokens = lr_blocks_tokens.view(num_lr, 4, token_dim)  # [num_lr, 4, 64]
                    
                    # Reshape token to patch format: [num_lr, 4, pF, pH, pW, C]
                    lr_patches = lr_blocks_tokens.view(num_lr, 4, pF, pH, pW, C)
                    
                    # Arrange 4 patches into 2x2 grid for each LR block
                    # patches 0,1,2,3 -> positions (0,0), (0,1), (1,0), (1,1) in the 2x2 token grid
                    lr_patches = lr_patches.view(num_lr, 2, 2, pF, pH, pW, C)
                    
                    # Unpatchify to spatial: [num_lr, pF, 2*pH, 2*pW, C] = [num_lr, 1, 4, 4, 16]
                    lr_spatial = lr_patches.permute(0, 3, 1, 4, 2, 5, 6)  # [num_lr, pF, 2, pH, 2, pW, C]
                    lr_spatial = lr_spatial.reshape(num_lr, pF, 2*pH, 2*pW, C)  # [num_lr, 1, 4, 4, 16]
                    
                    # Convert to [num_lr, C, H, W] format for downsampler: [num_lr, 16, 4, 4]
                    lr_spatial = lr_spatial.squeeze(1).permute(0, 3, 1, 2)  # [num_lr, 16, 4, 4]
                    
                    # Apply token_downsampler: [num_lr, 16, 4, 4] -> [num_lr, 16, 2, 2]
                    lr_downsampled = token_downsampler(
                        lr_spatial, 
                        temb=timestep,
                        foveation_mask=None  # Already processing only LR blocks
                    )
                    
                    # Repatchify to single token: [num_lr, 16, 2, 2] -> [num_lr, 64]
                    # [num_lr, C, 2, 2] -> [num_lr, pF, pH, pW, C] -> [num_lr, token_dim]
                    lr_downsampled = lr_downsampled.permute(0, 2, 3, 1)  # [num_lr, 2, 2, 16]
                    lr_downsampled = lr_downsampled.unsqueeze(1)  # [num_lr, 1(pF), 2, 2, 16]
                    lr_new_tokens = lr_downsampled.reshape(num_lr, token_dim)  # [num_lr, 64]
                    
                    # Replace top-left token (index 0) of each LR block with downsampled token
                    # image_blocks: [h_2, w_2, 2, 2, token_dim]
                    # We need to update position [0, 0] of each LR block
                    image_blocks_updated = image_blocks.clone()
                    image_blocks_updated[lr_indices[0], lr_indices[1], 0, 0, :] = lr_new_tokens
                    
                    # Reshape back to flat token sequence
                    image_patched = image_blocks_updated.permute(0, 2, 1, 3, 4).reshape(H_tokens * W_tokens, token_dim)

            # Expand for temporal dimension if needed
            if F_tokens > 1:
                valid_tokens = valid_tokens.unsqueeze(0).expand(F_tokens, -1).reshape(-1)
                res_mask = res_mask.unsqueeze(0).expand(F_tokens, -1).reshape(-1)
                tl_mask = tl_mask.unsqueeze(0).expand(F_tokens, -1).reshape(-1)

            # Filter to valid tokens
            image_filtered = image_patched[valid_tokens]
            pos_ids_hr_filtered = pos_ids_hr[valid_tokens]
            pos_ids_lr_filtered = pos_ids_lr[valid_tokens]
            res_mask_filtered = res_mask[valid_tokens]
            tl_mask_filtered = tl_mask[valid_tokens]

            # Add padding
            image_ori_len = len(image_filtered)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            if image_padding_len > 0:
                image_filtered = torch.cat([image_filtered, image_filtered[-1:].repeat(image_padding_len, 1)], dim=0)
                pos_ids_hr_filtered = torch.cat([pos_ids_hr_filtered, pos_ids_hr_filtered[-1:].repeat(image_padding_len, 1)], dim=0)
                pos_ids_lr_filtered = torch.cat([pos_ids_lr_filtered, pos_ids_lr_filtered[-1:].repeat(image_padding_len, 1)], dim=0)
                res_mask_filtered = torch.cat([res_mask_filtered, torch.zeros(image_padding_len, device=device)], dim=0)
                tl_mask_filtered = torch.cat([tl_mask_filtered, torch.zeros(image_padding_len, device=device, dtype=torch.bool)], dim=0)

            all_image_out.append(image_filtered)
            all_image_pos_ids_hr.append(pos_ids_hr_filtered)
            all_image_pos_ids_lr.append(pos_ids_lr_filtered)
            all_image_pad_mask.append(
                torch.cat([
                    torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                ], dim=0)
            )
            all_resolution_mask.append(res_mask_filtered)
            all_resolution_mask_top_left.append(tl_mask_filtered)

            # Store full sequence length, valid_tokens, and spatial dims for expansion during unpatchify
            all_valid_tokens.append(valid_tokens)
            all_full_seqlens.append(F_tokens * H_tokens * W_tokens)
            all_spatial_dims.append((F_tokens, H_tokens, W_tokens, h_2, w_2))
            all_is_low_res_block.append(is_low_res_block)

        return all_image_out, all_cap_feats_out, {
            "x_size": all_image_size,
            "x_pos_ids_hr": all_image_pos_ids_hr,
            "x_pos_ids_lr": all_image_pos_ids_lr,
            "cap_pos_ids": all_cap_pos_ids,
            "x_pad_mask": all_image_pad_mask,
            "cap_pad_mask": all_cap_pad_mask,
            "resolution_mask": all_resolution_mask,
            "resolution_mask_top_left": all_resolution_mask_top_left,
            "valid_tokens": all_valid_tokens,  # For expanding back to full res
            "full_seqlens": all_full_seqlens,  # Original full sequence lengths
            "spatial_dims": all_spatial_dims,  # (F_tokens, H_tokens, W_tokens, h_2, w_2)
            "is_low_res_block": all_is_low_res_block,  # Boolean mask for LR blocks [h_2, w_2]
        }

    def expand_foveated_tokens(
        self,
        x: List[torch.Tensor],
        valid_tokens: List[torch.Tensor],
        full_seqlens: List[int],
        spatial_dims: List[Tuple] = None,
        is_low_res_block: List[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Expand foveated (reduced) tokens back to full resolution.
        
        LR tokens (single token per 2x2 block) are duplicated to fill all 4 positions.
        The tokens are in raster scan order, not block order.
        
        Args:
            x: List of reduced token tensors [reduced_seq, dim]
            valid_tokens: List of boolean masks indicating valid positions in full sequence
            full_seqlens: Original full sequence lengths
            spatial_dims: List of (F_tokens, H_tokens, W_tokens, h_2, w_2) tuples
            is_low_res_block: List of boolean masks [h_2, w_2] indicating LR blocks
            
        Returns:
            List of expanded tensors [full_seq, dim]
        """
        expanded = []
        for i, (xi, valid, full_len) in enumerate(zip(x, valid_tokens, full_seqlens)):
            # Create output tensor
            dim = xi.shape[-1]
            device = xi.device
            dtype = xi.dtype
            
            # Get number of valid tokens (excluding padding)
            num_valid = valid.sum().item()
            
            # Create full-resolution output
            full_output = torch.zeros(full_len, dim, device=device, dtype=dtype)
            
            # Place valid tokens at their correct positions
            full_output[valid] = xi[:num_valid]
            
            # Now we need to fill in the missing LR block positions
            # LR blocks have only the top-left token (position 0 of each 2x2 block)
            # We need to copy this value to the other 3 positions in each LR block
            
            if spatial_dims is not None and is_low_res_block is not None:
                F_tokens, H_tokens, W_tokens, h_2, w_2 = spatial_dims[i]
                lr_block_mask = is_low_res_block[i]  # [h_2, w_2]
                
                # Reshape full_output to spatial format for easier manipulation
                # full_output: [F_tokens * H_tokens * W_tokens, dim]
                # Reshape to [F_tokens, H_tokens, W_tokens, dim]
                full_output_spatial = full_output.view(F_tokens, H_tokens, W_tokens, dim)
                
                # For each LR block (2x2 region), copy top-left to other positions
                # lr_block_mask is [h_2, w_2] - each True element represents a 2x2 block that's LR
                for bi in range(h_2):
                    for bj in range(w_2):
                        if lr_block_mask[bi, bj]:
                            # This 2x2 block is LR
                            # Pixel positions in the block: [2*bi:2*bi+2, 2*bj:2*bj+2]
                            r0, r1 = 2 * bi, 2 * bi + 2
                            c0, c1 = 2 * bj, 2 * bj + 2
                            
                            # Get top-left value
                            tl_value = full_output_spatial[:, r0, c0, :]  # [F_tokens, dim]
                            
                            # Copy to all 4 positions
                            full_output_spatial[:, r0, c0, :] = tl_value
                            full_output_spatial[:, r0, c1-1, :] = tl_value  # top-right
                            full_output_spatial[:, r1-1, c0, :] = tl_value  # bottom-left
                            full_output_spatial[:, r1-1, c1-1, :] = tl_value  # bottom-right
                
                full_output = full_output_spatial.view(full_len, dim)
            
            expanded.append(full_output)
        
        return expanded

    def unpatchify(
        self,
        x: List[torch.Tensor],
        size: List[Tuple],
        patch_size=2,
        f_patch_size=1,
        x_pos_offsets: Optional[List[Tuple[int, int]]] = None,
    ) -> List[torch.Tensor]:
        """Unpatchify tokens back to image format."""
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz

        if x_pos_offsets is not None:
            result = []
            for i in range(bsz):
                unified_x = x[i][x_pos_offsets[i][0]:x_pos_offsets[i][1]]
                cu_len = 0
                x_item = None
                for j in range(len(size[i])):
                    if size[i][j] is None:
                        ori_len = 0
                        pad_len = SEQ_MULTI_OF
                        cu_len += pad_len + ori_len
                    else:
                        F, H, W = size[i][j]
                        ori_len = (F // pF) * (H // pH) * (W // pW)
                        pad_len = (-ori_len) % SEQ_MULTI_OF
                        x_item = (
                            unified_x[cu_len:cu_len + ori_len]
                            .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                            .permute(6, 0, 3, 1, 4, 2, 5)
                            .reshape(self.out_channels, F, H, W)
                        )
                        cu_len += ori_len + pad_len
                result.append(x_item)
            return result
        else:
            for i in range(bsz):
                F, H, W = size[i]
                ori_len = (F // pF) * (H // pH) * (W // pW)
                x[i] = (
                    x[i][:ori_len]
                    .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                    .permute(6, 0, 3, 1, 4, 2, 5)
                    .reshape(self.out_channels, F, H, W)
                )
            return x

    def _prepare_sequence(
        self,
        feats: List[torch.Tensor],
        pos_ids: List[torch.Tensor],
        inner_pad_mask: List[torch.Tensor],
        pad_token: torch.nn.Parameter,
        noise_mask: Optional[List[List[int]]] = None,
        device: torch.device = None,
    ):
        """Prepare sequence with padding and RoPE."""
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)

        feats_cat = torch.cat(feats, dim=0)
        feats_cat[torch.cat(inner_pad_mask)] = pad_token.to(dtype=feats_cat.dtype, device=feats_cat.device)
        feats = list(feats_cat.split(item_seqlens, dim=0))

        freqs_cis = list(self.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0))

        feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis = pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, :feats.shape[1]]

        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(item_seqlens):
            attn_mask[i, :seq_len] = 1

        noise_mask_tensor = None
        if noise_mask is not None:
            noise_mask_tensor = pad_sequence(
                [torch.tensor(m, dtype=torch.long, device=device) for m in noise_mask],
                batch_first=True,
                padding_value=0,
            )[:, :feats.shape[1]]

        return feats, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor

    def _prepare_sequence_foveated(
        self,
        feats: List[torch.Tensor],
        pos_ids_hr: List[torch.Tensor],
        pos_ids_lr: List[torch.Tensor],
        inner_pad_mask: List[torch.Tensor],
        pad_token: torch.nn.Parameter,
        resolution_mask: List[torch.Tensor],
        resolution_mask_top_left: List[torch.Tensor],
        device: torch.device = None,
    ):
        """Prepare sequence with foveation support (dual RoPE)."""
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)

        # Apply pad token
        feats_cat = torch.cat(feats, dim=0)
        feats_cat[torch.cat(inner_pad_mask)] = pad_token.to(dtype=feats_cat.dtype, device=feats_cat.device)
        feats = list(feats_cat.split(item_seqlens, dim=0))

        # Compute dual RoPE
        freqs_cis_hr = list(self.rope_embedder(torch.cat(pos_ids_hr, dim=0)).split([len(p) for p in pos_ids_hr], dim=0))
        freqs_cis_lr = list(self.rope_embedder(torch.cat(pos_ids_lr, dim=0)).split([len(p) for p in pos_ids_lr], dim=0))

        # Pad to batch
        feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis_hr = pad_sequence(freqs_cis_hr, batch_first=True, padding_value=0.0)[:, :feats.shape[1]]
        freqs_cis_lr = pad_sequence(freqs_cis_lr, batch_first=True, padding_value=0.0)[:, :feats.shape[1]]

        # Attention mask
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(item_seqlens):
            attn_mask[i, :seq_len] = 1

        # Resolution masks
        res_mask = pad_sequence(resolution_mask, batch_first=True, padding_value=0.0)[:, :feats.shape[1]]
        res_mask_tl = pad_sequence(
            [m.float() for m in resolution_mask_top_left], 
            batch_first=True, 
            padding_value=0.0
        )[:, :feats.shape[1]].bool()

        return feats, freqs_cis_hr, freqs_cis_lr, attn_mask, item_seqlens, res_mask, res_mask_tl

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        siglip_feats=None,
        image_noise_mask=None,
        patch_size=2,
        f_patch_size=1,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        # Foveation parameters
        foveation_mask: torch.Tensor = None,
        token_downsampler: nn.Module = None,
    ):
        """
        Forward pass with optional foveation support.
        
        Args:
            x: List of image tensors [C, F, H, W] or nested list for omni mode
            t: Timestep tensor
            cap_feats: List of caption features
            siglip_feats: Optional SigLIP features
            image_noise_mask: Optional noise mask for omni mode
            patch_size: Spatial patch size
            f_patch_size: Temporal patch size
            use_gradient_checkpointing: Enable gradient checkpointing
            use_gradient_checkpointing_offload: Enable gradient checkpointing with offload
            foveation_mask: Optional foveation mask [H, W] where 1=HR, 0=LR
            token_downsampler: Optional module for spatial downsampling of LR regions
        """
        assert patch_size in self.all_patch_size and f_patch_size in self.all_f_patch_size
        omni_mode = isinstance(x[0], list)
        device = x[0][-1].device if omni_mode else x[0].device

        # Time embedding
        if omni_mode:
            t_noisy = self.t_embedder(t * self.t_scale).type_as(x[0][-1])
            t_clean = self.t_embedder(torch.ones_like(t) * self.t_scale).type_as(x[0][-1])
            adaln_input = None
        else:
            adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])
            t_noisy = t_clean = None

        # Determine if using foveation
        use_foveation = foveation_mask is not None and not omni_mode

        # Patchify
        if use_foveation:
            # foveation_mask: [128, 128]
            (x, cap_feats, patch_metadata) = self.patchify_and_embed_foveated(
                x, cap_feats, foveation_mask, patch_size, f_patch_size,
                token_downsampler=token_downsampler,
                timestep=t,
            )
            x_pos_ids_hr = patch_metadata["x_pos_ids_hr"]
            x_pos_ids_lr = patch_metadata["x_pos_ids_lr"]
            x_resolution_mask = patch_metadata["resolution_mask"]
            x_resolution_mask_tl = patch_metadata["resolution_mask_top_left"]
        else:
            (x, cap_feats, patch_metadata) = self.patchify_and_embed(
                x, cap_feats, patch_size, f_patch_size
            )

        x_size = patch_metadata["x_size"]
        x_pos_ids = patch_metadata.get("x_pos_ids", patch_metadata.get("x_pos_ids_hr"))
        cap_pos_ids = patch_metadata["cap_pos_ids"]
        x_pad_mask = patch_metadata["x_pad_mask"]
        cap_pad_mask = patch_metadata["cap_pad_mask"]

        # Embed and refine x
        x_seqlens = [len(xi) for xi in x]
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x, dim=0))

        if use_foveation:
            x, x_freqs_hr, x_freqs_lr, x_mask, _, x_res_mask, x_res_mask_tl = self._prepare_sequence_foveated(
                list(x.split(x_seqlens, dim=0)),
                x_pos_ids_hr, x_pos_ids_lr,
                x_pad_mask, self.x_pad_token,
                x_resolution_mask, x_resolution_mask_tl,
                device
            )
            x_freqs = None  # Not used in foveated mode
        else:
            x, x_freqs, x_mask, _, x_noise_tensor = self._prepare_sequence(
                list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, self.x_pad_token, None, device
            )
            x_freqs_hr = x_freqs_lr = x_res_mask = x_res_mask_tl = None

        # Noise refiner
        for layer in self.noise_refiner:
            x = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=x, attn_mask=x_mask, freqs_cis=x_freqs, adaln_input=adaln_input,
                freqs_cis_hr=x_freqs_hr, freqs_cis_lr=x_freqs_lr,
                resolution_mask=x_res_mask[0] if x_res_mask is not None else None,
                resolution_mask_top_left=x_res_mask_tl[0] if x_res_mask_tl is not None else None,
            )

        # Cap embed & refine
        cap_seqlens = [len(ci) for ci in cap_feats]
        cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))
        cap_feats, cap_freqs, cap_mask, _, _ = self._prepare_sequence(
            list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, self.cap_pad_token, None, device
        )

        for layer in self.context_refiner:
            cap_feats = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=cap_feats, attn_mask=cap_mask, freqs_cis=cap_freqs,
            )

        # Build unified sequence [x, cap]
        bsz = len(x_seqlens)
        unified = []
        unified_freqs = []
        unified_freqs_hr = [] if use_foveation else None
        unified_freqs_lr = [] if use_foveation else None
        unified_res_mask = [] if use_foveation else None
        unified_res_mask_tl = [] if use_foveation else None

        for i in range(bsz):
            x_len, cap_len = x_seqlens[i], cap_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            
            if use_foveation:
                unified_freqs_hr.append(torch.cat([x_freqs_hr[i][:x_len], cap_freqs[i][:cap_len]]))
                unified_freqs_lr.append(torch.cat([x_freqs_lr[i][:x_len], cap_freqs[i][:cap_len]]))
                # Caption tokens are treated as HR (resolution_mask = 1)
                cap_res_mask = torch.ones(cap_len, device=device)
                cap_res_mask_tl = torch.ones(cap_len, device=device, dtype=torch.bool)
                unified_res_mask.append(torch.cat([x_res_mask[i][:x_len], cap_res_mask]))
                unified_res_mask_tl.append(torch.cat([x_res_mask_tl[i][:x_len], cap_res_mask_tl]))
            else:
                unified_freqs.append(torch.cat([x_freqs[i][:x_len], cap_freqs[i][:cap_len]]))

        unified_seqlens = [a + b for a, b in zip(x_seqlens, cap_seqlens)]
        max_seqlen = max(unified_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        
        if use_foveation:
            unified_freqs_hr = pad_sequence(unified_freqs_hr, batch_first=True, padding_value=0.0)
            unified_freqs_lr = pad_sequence(unified_freqs_lr, batch_first=True, padding_value=0.0)
            unified_res_mask = pad_sequence(unified_res_mask, batch_first=True, padding_value=0.0)
            unified_res_mask_tl = pad_sequence(
                [m.float() for m in unified_res_mask_tl], 
                batch_first=True, 
                padding_value=0.0
            ).bool()
            unified_freqs = None
        else:
            unified_freqs = pad_sequence(unified_freqs, batch_first=True, padding_value=0.0)

        unified_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_seqlens):
            unified_mask[i, :seq_len] = 1

        # Main transformer layers
        for layer_idx, layer in enumerate(self.layers):
            unified = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=unified, attn_mask=unified_mask, freqs_cis=unified_freqs, adaln_input=adaln_input,
                freqs_cis_hr=unified_freqs_hr[0] if unified_freqs_hr is not None else None,
                freqs_cis_lr=unified_freqs_lr[0] if unified_freqs_lr is not None else None,
                resolution_mask=unified_res_mask[0] if unified_res_mask is not None else None,
                resolution_mask_top_left=unified_res_mask_tl[0] if unified_res_mask_tl is not None else None,
            )

        # Final layer
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)

        # Unpatchify - extract x portion
        x_outputs = []
        for i in range(bsz):
            x_len = x_seqlens[i]
            x_outputs.append(unified[i][:x_len])

        # If foveation was used, expand reduced tokens back to full resolution
        if use_foveation:
            valid_tokens = patch_metadata.get("valid_tokens")
            full_seqlens = patch_metadata.get("full_seqlens")
            spatial_dims = patch_metadata.get("spatial_dims")
            is_low_res_block = patch_metadata.get("is_low_res_block")
            if valid_tokens is not None and full_seqlens is not None:
                x_outputs = self.expand_foveated_tokens(
                    x_outputs, valid_tokens, full_seqlens,
                    spatial_dims=spatial_dims,
                    is_low_res_block=is_low_res_block
                )

        x = self.unpatchify(x_outputs, x_size, patch_size, f_patch_size)

        return x
