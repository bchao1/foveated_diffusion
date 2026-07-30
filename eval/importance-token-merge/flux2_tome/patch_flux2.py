"""
ToMe-style token merging for FLUX.2, applied via subclassing.

Design notes
------------
* DiffSynth is never edited or mutated. We define subclasses of `Flux2DiT` and the two
  block types here, and `apply_patch` rebinds the *instance's* `__class__` to ours. The
  original classes are untouched, and `remove_patch` restores them. The foveated pipeline
  and foveated DiT are never imported.

* Unlike upstream ToMe, the grouping is fixed by the foveation mask, so it does not depend
  on per-layer features. We therefore build it once at patch time and compute the merged
  RoPE table once per DiT forward, instead of re-deriving a matching in every block.

* RoPE is why block-only patching is not enough. FLUX.2 builds `concat_rotary_emb` at full
  [txt + img] length in `Flux2DiT.forward` and applies it inside attention, so changing the
  image sequence length inside a block requires a correspondingly gathered rope table. That
  gather is the reason `Flux2DiT.forward` is overridden at all.

* `merge_mlp` only has an independent meaning in the 8 double-stream blocks. The 48
  single-stream blocks are *parallel* blocks: `to_qkv_mlp_proj` computes the attention QKV
  and the MLP input in one fused projection, so feeding them merged tokens necessarily
  merges the MLP too. There is no way to merge attention but not the MLP there short of
  running the block twice. Both configs therefore share identical behavior in 48/56 blocks.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from diffsynth.models.flux2_dit import (
    Flux2DiT,
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
)
from diffsynth.core.gradient import gradient_checkpoint_forward

from .merge_2x2 import BlockMerger


class ToMeFlux2TransformerBlock(Flux2TransformerBlock):
    """Double-stream block. Image tokens are merged around attention (and optionally the
    image MLP); text tokens always pass through at full length."""

    _tome_parent = Flux2TransformerBlock

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_params_img,
        temb_mod_params_txt,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self._tome_state
        if not state.get("enabled"):
            return self._tome_parent.forward(
                self, hidden_states, encoder_hidden_states, temb_mod_params_img,
                temb_mod_params_txt, image_rotary_emb, joint_attention_kwargs,
            )

        merger: BlockMerger = state["merger"]
        joint_attention_kwargs = joint_attention_kwargs or {}

        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

        # Img stream
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        # Conditioning txt stream
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        # --- ToMe: merge image tokens, attend with the gathered rope, unmerge ---
        attn_output, context_attn_output = self.attn(
            hidden_states=merger.merge(norm_hidden_states),
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=state["merged_rope"],
            **joint_attention_kwargs,
        )
        attn_output = merger.unmerge(attn_output)

        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if state.get("merge_mlp"):
            ff_output = merger.unmerge(self.ff(merger.merge(norm_hidden_states)))
        else:
            ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        # Txt stream (never merged)
        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class ToMeFlux2SingleTransformerBlock(Flux2SingleTransformerBlock):
    """Single-stream (parallel) block. `hidden_states` arrives as [txt | img]; only the
    image slice is merged. Attention and MLP share one fused projection, so both are
    merged regardless of the `merge_mlp` setting."""

    _tome_parent = Flux2SingleTransformerBlock

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb_mod_params,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        split_hidden_states: bool = False,
        text_seq_len: Optional[int] = None,
    ):
        state = self._tome_state
        if not state.get("enabled"):
            return self._tome_parent.forward(
                self, hidden_states, encoder_hidden_states, temb_mod_params,
                image_rotary_emb, joint_attention_kwargs, split_hidden_states, text_seq_len,
            )

        merger: BlockMerger = state["merger"]

        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        if text_seq_len is None:
            text_seq_len = state["text_seq_len"]

        mod_shift, mod_scale, mod_gate = temb_mod_params

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        # --- ToMe: merge the image slice only ---
        norm_txt = norm_hidden_states[:, :text_seq_len]
        norm_img = merger.merge(norm_hidden_states[:, text_seq_len:])

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=torch.cat([norm_txt, norm_img], dim=1),
            image_rotary_emb=state["merged_rope"],
            **joint_attention_kwargs,
        )
        attn_output = torch.cat(
            [attn_output[:, :text_seq_len], merger.unmerge(attn_output[:, text_seq_len:])], dim=1
        )

        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if split_hidden_states:
            return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return hidden_states


class ToMeFlux2DiT(Flux2DiT):
    """Mirrors `Flux2DiT.forward`, adding only the merged-RoPE gather and the shared
    per-forward state the patched blocks read."""

    _tome_parent = Flux2DiT

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
    ):
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            joint_attention_kwargs.pop("scale", 1.0)

        num_txt_tokens = encoder_hidden_states.shape[1]

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)[0]

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # --- ToMe: rope table for the merged sequence, built once per forward ---
        state = self._tome_state
        if state.get("enabled"):
            merger: BlockMerger = state["merger"]
            if img_ids.shape[0] != merger.num_tokens:
                raise ValueError(
                    f"merger built for {merger.num_tokens} image tokens but got {img_ids.shape[0]}; "
                    "the mask/resolution passed to apply_patch does not match this run"
                )
            state["merged_rope"] = (
                torch.cat([text_rotary_emb[0], merger.gather_positions(image_rotary_emb[0])], dim=0),
                torch.cat([text_rotary_emb[1], merger.gather_positions(image_rotary_emb[1])], dim=0),
            )
            state["text_seq_len"] = num_txt_tokens

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for block in self.single_transformer_blocks:
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

        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


def apply_patch(
    dit: Flux2DiT,
    foveation_mask: torch.Tensor,
    latent_height: int,
    latent_width: int,
    lr_factor: int = 2,
    merge_mlp: bool = False,
    enabled: bool = True,
):
    """Rebind `dit` and its blocks to the ToMe subclasses. Returns the shared state dict.

    Call *after* any LoRA / checkpoint loading; class rebinding does not disturb weights.
    """
    remove_patch(dit)

    merger = BlockMerger(foveation_mask, latent_height, latent_width, lr_factor=lr_factor)
    state = {
        "enabled": enabled,
        "merge_mlp": merge_mlp,
        "merger": merger,
        "merged_rope": None,
        "text_seq_len": None,
    }

    dit._tome_state = state
    dit.__class__ = ToMeFlux2DiT
    for block in dit.transformer_blocks:
        block._tome_state = state
        block.__class__ = ToMeFlux2TransformerBlock
    for block in dit.single_transformer_blocks:
        block._tome_state = state
        block.__class__ = ToMeFlux2SingleTransformerBlock

    print(
        f"[flux2_tome] {merger.num_tokens} -> {merger.num_merged} image tokens "
        f"({merger.num_merged / merger.num_tokens:.1%} kept), lr_factor={lr_factor}, "
        f"merge_mlp={merge_mlp} (double blocks only; single blocks always merge both)"
    )
    return state


def remove_patch(dit: torch.nn.Module):
    """Restore the original classes."""
    for module in [dit, *getattr(dit, "transformer_blocks", []), *getattr(dit, "single_transformer_blocks", [])]:
        if hasattr(module.__class__, "_tome_parent"):
            module.__class__ = module.__class__._tome_parent
        if hasattr(module, "_tome_state"):
            del module._tome_state
    return dit
