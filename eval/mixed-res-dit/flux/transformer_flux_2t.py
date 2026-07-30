from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from .transformer_flux import FluxTransformer2DModel as FluxTransformer2DModel_BASE

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class FluxTransformer2DModel(FluxTransformer2DModel_BASE):
    @staticmethod
    def _mix_by_mask(param_lr: torch.Tensor, param_hr: torch.Tensor, image_is_lr: torch.Tensor) -> torch.Tensor:
        # param_*: [B, D], image_is_lr: [B, N, 1] -> [B, N, D]
        return image_is_lr * param_lr[:, None, :] + (1.0 - image_is_lr) * param_hr[:, None, :]

    def _get_temporal_cond(
        self,
        timestep: torch.Tensor,
        pooled_projections: torch.Tensor,
        hidden_dtype: torch.dtype,
        guidance: Optional[torch.Tensor],
    ) -> torch.Tensor:
        timestep = timestep.to(hidden_dtype) * 1000
        if guidance is None:
            return self.time_text_embed(timestep, pooled_projections)
        guidance = guidance.to(hidden_dtype) * 1000
        return self.time_text_embed(timestep, guidance, pooled_projections)

    @staticmethod
    def _adaln_continuous_mixed(norm_mod, x: torch.Tensor, temb_lr: torch.Tensor, temb_hr: torch.Tensor, image_is_lr: torch.Tensor) -> torch.Tensor:
        emb_lr = norm_mod.linear(norm_mod.silu(temb_lr).to(x.dtype))
        emb_hr = norm_mod.linear(norm_mod.silu(temb_hr).to(x.dtype))
        scale_lr, shift_lr = torch.chunk(emb_lr, 2, dim=1)
        scale_hr, shift_hr = torch.chunk(emb_hr, 2, dim=1)

        scale = image_is_lr * scale_lr[:, None, :] + (1.0 - image_is_lr) * scale_hr[:, None, :]
        shift = image_is_lr * shift_lr[:, None, :] + (1.0 - image_is_lr) * shift_hr[:, None, :]
        return norm_mod.norm(x) * (1.0 + scale) + shift

    def forward_mixed(
        self,
        hidden_states_lr: torch.Tensor,
        hidden_states_hr: torch.Tensor,
        mask_lr_tokens: torch.Tensor = None,
        mask_lr_keep_tokens: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep_lr: torch.LongTensor = None,
        timestep_hr: torch.LongTensor = None,
        img_ids_lr: torch.Tensor = None,
        img_ids_hr: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if timestep_hr is None:
            timestep_hr = timestep_lr

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
            joint_attention_kwargs = {}

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        b = hidden_states_lr.shape[0]
        b_hr = hidden_states_hr.shape[0]
        b_m, _, ht, wt = mask_lr_tokens.shape
        assert b == b_hr and b == b_m

        # Compute linear-embedded tokens for LR and HR
        tokens_lr_flat = self.x_embedder(hidden_states_lr)
        tokens_hr_flat = self.x_embedder(hidden_states_hr)

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

        mixed_tokens = torch.stack(mixed_tokens_list, dim=0)

        rotary_emb_lr = mask_state["rotary_emb_lr"]
        rotary_emb_hr_query = mask_state["rotary_emb_hr_query"]
        lr_overlap_idx_rev = mask_state["lr_overlap_idx_rev"]
        hr_overlap_idx_rev = mask_state["hr_overlap_idx_rev"]
        has_overlap_hr = mask_state["has_overlap_hr"]
        has_overlap_lr = mask_state["has_overlap_lr"]

        # Build two global temporal conditions; later we only blend modulation params by token type.
        temb_lr = self._get_temporal_cond(timestep_lr, pooled_projections, hidden_states_lr.dtype, guidance)
        temb_hr = self._get_temporal_cond(timestep_hr, pooled_projections, hidden_states_lr.dtype, guidance)
        # Keep text branch stable with LR time embedding.
        temb_text = temb_lr

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        assert txt_ids.ndim == 2
        rotary_txt = self.pos_embed(txt_ids)

        if "ip_adapter_image_embeds" in joint_attention_kwargs:
            raise AssertionError("forward_mixed does not support ip_adapter_image_embeds in joint_attention_kwargs")

        hidden_states = mixed_tokens

        if n_lr == N or not self.seperate_attention:
            token_info = None
            if self.base_use_rope_lr:
                image_rotary_emb = (
                    torch.cat([rotary_txt[0], rotary_emb_lr[0]]),
                    torch.cat([rotary_txt[1], rotary_emb_lr[1]]),
                )
            else:
                image_rotary_emb = (
                    torch.cat([rotary_txt[0], rotary_emb_hr_query[0]]),
                    torch.cat([rotary_txt[1], rotary_emb_hr_query[1]]),
                )
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

        # In mixed_tokens, first n_lr image tokens are LR and the rest are HR.
        image_is_lr = hidden_states.new_ones((b, N, 1))
        if n_lr < N:
            image_is_lr[:, n_lr:, :] = 0.0

        for block in self.transformer_blocks:
            # image branch: blend AdaLN modulation params per token, then run shared attention/FF once.
            emb_lr = block.norm1.linear(block.norm1.silu(temb_lr))
            emb_hr = block.norm1.linear(block.norm1.silu(temb_hr))
            shift_msa_lr, scale_msa_lr, gate_msa_lr, shift_mlp_lr, scale_mlp_lr, gate_mlp_lr = emb_lr.chunk(6, dim=1)
            shift_msa_hr, scale_msa_hr, gate_msa_hr, shift_mlp_hr, scale_mlp_hr, gate_mlp_hr = emb_hr.chunk(6, dim=1)

            shift_msa = self._mix_by_mask(shift_msa_lr, shift_msa_hr, image_is_lr)
            scale_msa = self._mix_by_mask(scale_msa_lr, scale_msa_hr, image_is_lr)
            gate_msa = self._mix_by_mask(gate_msa_lr, gate_msa_hr, image_is_lr)
            shift_mlp = self._mix_by_mask(shift_mlp_lr, shift_mlp_hr, image_is_lr)
            scale_mlp = self._mix_by_mask(scale_mlp_lr, scale_mlp_hr, image_is_lr)
            gate_mlp = self._mix_by_mask(gate_mlp_lr, gate_mlp_hr, image_is_lr)

            norm_hidden_states = block.norm1.norm(hidden_states)
            norm_hidden_states = norm_hidden_states * (1.0 + scale_msa) + shift_msa

            # text/context branch uses text temporal embedding.
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = block.norm1_context(
                encoder_hidden_states, emb=temb_text
            )

            attention_outputs = block.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                token_info=token_info,
                **joint_attention_kwargs,
            )

            if len(attention_outputs) == 2:
                attn_output, context_attn_output = attention_outputs
                ip_attn_output = None
            elif len(attention_outputs) == 3:
                attn_output, context_attn_output, ip_attn_output = attention_outputs
            else:
                raise RuntimeError("Unexpected attention output tuple length.")

            attn_output = gate_msa * attn_output
            hidden_states = hidden_states + attn_output

            norm_hidden_states = block.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1.0 + scale_mlp) + shift_mlp
            ff_output = block.ff(norm_hidden_states)
            ff_output = gate_mlp * ff_output
            hidden_states = hidden_states + ff_output
            if ip_attn_output is not None:
                hidden_states = hidden_states + ip_attn_output

            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = block.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1.0 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            context_ff_output = block.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        for block in self.single_transformer_blocks:
            text_seq_len = encoder_hidden_states.shape[1]
            cat_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            residual = cat_hidden_states

            # token-wise AdaLN modulation for concat(text + image) sequence
            emb_text = block.norm.linear(block.norm.silu(temb_text))
            shift_text, scale_text, gate_text = emb_text.chunk(3, dim=1)

            emb_lr = block.norm.linear(block.norm.silu(temb_lr))
            emb_hr = block.norm.linear(block.norm.silu(temb_hr))
            shift_lr, scale_lr, gate_lr = emb_lr.chunk(3, dim=1)
            shift_hr, scale_hr, gate_hr = emb_hr.chunk(3, dim=1)

            shift_img = self._mix_by_mask(shift_lr, shift_hr, image_is_lr)
            scale_img = self._mix_by_mask(scale_lr, scale_hr, image_is_lr)
            gate_img = self._mix_by_mask(gate_lr, gate_hr, image_is_lr)

            shift = torch.cat([shift_text[:, None, :].expand(-1, text_seq_len, -1), shift_img], dim=1)
            scale = torch.cat([scale_text[:, None, :].expand(-1, text_seq_len, -1), scale_img], dim=1)
            gate = torch.cat([gate_text[:, None, :].expand(-1, text_seq_len, -1), gate_img], dim=1)

            norm_hidden_states = block.norm.norm(cat_hidden_states)
            norm_hidden_states = norm_hidden_states * (1.0 + scale) + shift

            mlp_hidden_states = block.act_mlp(block.proj_mlp(norm_hidden_states))
            attn_output = block.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                token_info=token_info,
                **joint_attention_kwargs,
            )

            cat_hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            cat_hidden_states = gate * block.proj_out(cat_hidden_states)
            cat_hidden_states = residual + cat_hidden_states

            if cat_hidden_states.dtype == torch.float16:
                cat_hidden_states = cat_hidden_states.clip(-65504, 65504)

            encoder_hidden_states, hidden_states = cat_hidden_states[:, :text_seq_len], cat_hidden_states[:, text_seq_len:]

        hidden_states = self._adaln_continuous_mixed(self.norm_out, hidden_states, temb_lr, temb_hr, image_is_lr)
        hidden_states = self.proj_out(hidden_states)

        new_lr = hidden_states_lr.clone()
        new_hr = hidden_states_hr.clone()
        for bidx in range(b):
            new_lr[bidx].index_copy_(0, lr_indices[bidx], hidden_states[bidx, :n_lr])
            if n_lr < N:
                new_hr[bidx].index_copy_(0, hr_indices[bidx], hidden_states[bidx, n_lr:])

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        return new_lr, new_hr
