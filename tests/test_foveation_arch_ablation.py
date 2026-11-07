"""Unit tests for the ICLR 2027 rebuttal architecture ablations of Flux2DiTFoveated.

    rope_mode="standard"     no phase-aligned RoPE: joint attention on HR-grid ids
    resolution_embedding     learned, zero-init HR/LR embedding after x_embedder

Runs on CPU with a tiny DiT:
    PYTHONPATH=DiffSynth-Studio python -m pytest -q tests/test_foveation_arch_ablation.py
"""
import os
import sys
import types

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "DiffSynth-Studio"))

import importlib  # noqa: E402

attention_mod = importlib.import_module("diffsynth.core.attention.attention")
from diffsynth.models.flux2_dit_foveated import (  # noqa: E402
    Flux2DiTFoveated,
    RES_EMBED_KEY,
    configure_foveation_arch_from_checkpoint,
    foveation_arch_metadata,
    read_foveation_arch,
)
from diffsynth.pipelines.flux2_image_foveated import Flux2FoveatedImagePipeline  # noqa: E402

attention_mod.ATTENTION_IMPLEMENTATION = "torch"  # CPU: plain SDPA

H = W = 8          # latent token grid
C_IN, TXT, TXT_DIM = 8, 5, 16


def tiny_dit(seed=0):
    torch.manual_seed(seed)
    dit = Flux2DiTFoveated(
        in_channels=C_IN, num_layers=1, num_single_layers=1, attention_head_dim=128,
        num_attention_heads=2, joint_attention_dim=TXT_DIM, guidance_embeds=False,
    )
    return dit.eval()


def inputs(with_mask=True, seed=1):
    g = torch.Generator().manual_seed(seed)
    mask = None
    if with_mask:
        mask = torch.zeros(H, W)
        mask[2:6, 2:6] = 1.0  # 4x4 fovea -> 4 HR blocks, 12 LR blocks
    ids, res, res_tl = Flux2FoveatedImagePipeline._prepare_latent_image_ids_foveation(
        1, H, W, torch.device("cpu"), torch.float32, mask, lr_factor=2)
    txt_ids = torch.zeros(TXT, 4)
    txt_ids[:, 3] = torch.arange(TXT)
    return dict(
        hidden_states=torch.randn(1, ids.shape[0], C_IN, generator=g),
        encoder_hidden_states=torch.randn(1, TXT, TXT_DIM, generator=g),
        timestep=torch.tensor([0.5]),
        img_ids=ids, txt_ids=txt_ids,
        resolution_mask=res, resolution_mask_top_left=res_tl,
    )


def run(dit, kw):
    with torch.no_grad():
        return dit(**kw)


def test_mask_has_both_resolutions():
    kw = inputs()
    n_hr = int(kw["resolution_mask"].sum())
    assert n_hr == 16 and kw["resolution_mask"].numel() == 16 + 12


def test_zero_init_embedding_is_identity():
    kw = inputs()
    ref = run(tiny_dit(), kw)
    dit = tiny_dit()
    dit.enable_resolution_embedding()
    assert torch.count_nonzero(dit.resolution_embedding.weight) == 0
    assert torch.equal(run(dit, kw), ref)


def test_standard_rope_is_plain_joint_attention():
    """standard + mask must equal the processors' no-mask branch on the same ids."""
    kw = inputs()
    dit = tiny_dit()
    dit.set_rope_mode("standard")
    out_std = run(dit, kw)
    out_plain = run(dit, {**kw, "resolution_mask": None, "resolution_mask_top_left": None})
    assert torch.equal(out_std, out_plain)


def test_phase_aligned_differs_from_standard():
    kw = inputs()
    dit = tiny_dit()
    out_pa = run(dit, kw)
    dit.set_rope_mode("standard")
    assert not torch.allclose(out_pa, run(dit, kw))


def test_rope_mode_without_mask_is_unchanged():
    """Full-resolution generation (no mask) is identical in both modes."""
    kw = inputs(with_mask=False)
    dit = tiny_dit()
    ref = run(dit, kw)
    dit.set_rope_mode("standard")
    assert torch.equal(run(dit, kw), ref)


def test_bad_rope_mode_raises():
    with pytest.raises(ValueError):
        tiny_dit().set_rope_mode("none")


def test_embedding_is_used_and_gets_gradient():
    kw = inputs()
    dit = tiny_dit()
    dit.enable_resolution_embedding()
    ref = run(dit, kw)
    out = dit(**kw)
    out.square().mean().backward()
    grad = dit.resolution_embedding.weight.grad
    assert grad is not None and grad[0].abs().sum() > 0 and grad[1].abs().sum() > 0
    with torch.no_grad():
        dit.resolution_embedding.weight.normal_()
    assert not torch.allclose(run(dit, kw), ref)


def test_embedding_survives_lora_injection_and_export():
    """LoRA injection freezes non-adapter params; train_foveated.py re-enables the embedding."""
    from diffsynth.diffusion.training_module import DiffusionTrainingModule

    class M(DiffusionTrainingModule):
        def __init__(self):
            super().__init__()
            self.pipe = torch.nn.Module()
            self.pipe.dit = tiny_dit()
            self.pipe.dit.enable_resolution_embedding()
            self.pipe.dit.requires_grad_(True)
            self.pipe.dit = self.add_lora_to_model(self.pipe.dit, target_modules=["to_q", "to_k"], lora_rank=4)
            self.pipe.dit.resolution_embedding.requires_grad_(True)

    m = M()
    sd = m.export_trainable_state_dict(m.state_dict(), remove_prefix="pipe.dit.")
    assert RES_EMBED_KEY in sd
    assert any("lora_A" in k for k in sd)
    assert not any(k.startswith("x_embedder") for k in sd)


def _save_ckpt(tmp_path, dit, name="ckpt.safetensors", with_metadata=True):
    from safetensors.torch import save_file
    sd = {"transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.zeros(4, 256)}
    if dit.resolution_embedding is not None:
        sd[RES_EMBED_KEY] = dit.resolution_embedding.weight.detach().clone()
    path = str(tmp_path / name)
    save_file(sd, path, metadata=foveation_arch_metadata(dit) if with_metadata else None)
    return path


@pytest.mark.parametrize("rope_mode,res_embed", [
    ("phase_aligned", False), ("standard", False), ("phase_aligned", True),
])
def test_checkpoint_roundtrip(tmp_path, rope_mode, res_embed):
    src = tiny_dit()
    src.set_rope_mode(rope_mode)
    if res_embed:
        src.enable_resolution_embedding()
        with torch.no_grad():
            src.resolution_embedding.weight.normal_()
    path = _save_ckpt(tmp_path, src)

    dst = tiny_dit()
    configure_foveation_arch_from_checkpoint(dst, path)
    assert dst.rope_mode == rope_mode
    assert (dst.resolution_embedding is not None) == res_embed
    if res_embed:
        assert torch.equal(dst.resolution_embedding.weight, src.resolution_embedding.weight)
    kw = inputs()
    assert torch.equal(run(dst, kw), run(src, kw))


def test_cli_conflicting_with_metadata_raises(tmp_path):
    src = tiny_dit()
    src.set_rope_mode("standard")
    path = _save_ckpt(tmp_path, src)
    with pytest.raises(ValueError):
        configure_foveation_arch_from_checkpoint(tiny_dit(), path, rope_mode="phase_aligned")
    with pytest.raises(ValueError):
        configure_foveation_arch_from_checkpoint(tiny_dit(), path, resolution_embedding=True)
    # Agreeing CLI values are fine.
    configure_foveation_arch_from_checkpoint(tiny_dit(), path, rope_mode="standard", resolution_embedding=False)


def test_old_checkpoint_loads_as_paper_model(tmp_path):
    path = _save_ckpt(tmp_path, tiny_dit(), with_metadata=False)
    arch = read_foveation_arch(path)
    assert arch["has_metadata"] is False and arch["rope_mode"] == "phase_aligned"
    dst = tiny_dit()
    configure_foveation_arch_from_checkpoint(dst, path)
    assert dst.rope_mode == "phase_aligned" and dst.resolution_embedding is None
    # CLI can fill in a checkpoint without metadata ...
    dst = tiny_dit()
    configure_foveation_arch_from_checkpoint(dst, path, rope_mode="standard")
    assert dst.rope_mode == "standard"
    # ... but cannot invent an embedding the checkpoint does not hold.
    with pytest.raises(ValueError):
        configure_foveation_arch_from_checkpoint(tiny_dit(), path, resolution_embedding=True)


def test_logger_writes_arch_metadata(tmp_path):
    """ModelLogger.save_model writes checkpoint_metadata() into the safetensors header."""
    from diffsynth.diffusion.logger import ModelLogger

    dit = tiny_dit()
    dit.set_rope_mode("standard")
    dit.enable_resolution_embedding()

    class Unwrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(2))

        def checkpoint_metadata(self):
            return foveation_arch_metadata(dit)

        def export_trainable_state_dict(self, sd, remove_prefix=None):
            return {RES_EMBED_KEY: dit.resolution_embedding.weight.detach()}

    m = Unwrapped()
    acc = types.SimpleNamespace(
        wait_for_everyone=lambda: None, is_main_process=True,
        unwrap_model=lambda x: x, get_state_dict=lambda x: x.state_dict(),
    )
    ModelLogger(str(tmp_path)).save_model(acc, m, "step-1.safetensors")
    arch = read_foveation_arch(str(tmp_path / "step-1.safetensors"))
    assert arch["has_metadata"] and arch["rope_mode"] == "standard" and arch["resolution_embedding"]
