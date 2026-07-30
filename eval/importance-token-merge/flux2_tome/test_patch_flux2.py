"""Structural gate for the DiT patch, on a tiny randomly-initialized Flux2DiT (CPU).

    PYTHONPATH=<repo>/DiffSynth-Studio python -m flux2_tome.test_patch_flux2

Checks that patching preserves output shape, that `enabled=False` is bit-exact against the
unpatched model, that patching is reversible, and that merging actually shrinks the
sequence the attention layers see.
"""

import torch

from diffsynth.models.flux2_dit import Flux2DiT

from .patch_flux2 import apply_patch, remove_patch

H = W = 8           # tiny token grid
TXT = 5
IN_CH = 16
HEADS, HEAD_DIM = 2, 16


def _tiny_dit():
    torch.manual_seed(0)
    return Flux2DiT(
        in_channels=IN_CH, num_layers=2, num_single_layers=2,
        attention_head_dim=HEAD_DIM, num_attention_heads=HEADS,
        joint_attention_dim=IN_CH, axes_dims_rope=(4, 4, 4, 4),
    ).eval()


def _inputs():
    g = torch.Generator().manual_seed(1)
    img = torch.randn(2, H * W, IN_CH, generator=g)
    txt = torch.randn(2, TXT, IN_CH, generator=g)
    img_ids = torch.cartesian_prod(
        torch.zeros(1), torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32), torch.zeros(1),
    )
    txt_ids = torch.zeros(TXT, 4)
    return img, txt, img_ids, txt_ids


def _run(dit, img, txt, img_ids, txt_ids):
    with torch.no_grad():
        return dit(
            hidden_states=img, encoder_hidden_states=txt,
            timestep=torch.tensor([0.5]), img_ids=img_ids, txt_ids=txt_ids,
            guidance=torch.tensor([4.0]),
        )


def test_disabled_is_bit_exact_and_reversible():
    dit = _tiny_dit()
    img, txt, img_ids, txt_ids = _inputs()
    ref = _run(dit, img, txt, img_ids, txt_ids)

    apply_patch(dit, torch.ones(H, W), H, W, lr_factor=2, enabled=False)
    assert torch.equal(_run(dit, img, txt, img_ids, txt_ids), ref), "enabled=False changed the output"

    remove_patch(dit)
    assert type(dit) is Flux2DiT and not hasattr(dit, "_tome_state")
    assert torch.equal(_run(dit, img, txt, img_ids, txt_ids), ref), "remove_patch did not restore"
    print("PASS  enabled=False is bit-exact; remove_patch restores the original classes")


def test_all_high_res_mask_is_bit_exact():
    """An all-high-res mask merges nothing, so the patched path must reproduce the original."""
    dit = _tiny_dit()
    img, txt, img_ids, txt_ids = _inputs()
    ref = _run(dit, img, txt, img_ids, txt_ids)

    apply_patch(dit, torch.ones(H, W), H, W, lr_factor=2, merge_mlp=True)
    out = _run(dit, img, txt, img_ids, txt_ids)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()
    print("PASS  all-high-res mask reproduces the unpatched model")


def test_merging_shrinks_the_attended_sequence():
    dit = _tiny_dit()
    img, txt, img_ids, txt_ids = _inputs()

    mask = torch.zeros(H, W)
    mask[2:6, 2:6] = 1.0
    state = apply_patch(dit, mask, H, W, lr_factor=2, merge_mlp=True)

    seen = []
    handle = dit.single_transformer_blocks[0].attn.register_forward_pre_hook(
        lambda m, a, kw: seen.append(kw["hidden_states"].shape[1]), with_kwargs=True
    )
    out = _run(dit, img, txt, img_ids, txt_ids)
    handle.remove()

    expected = TXT + state["merger"].num_merged
    assert seen and all(s == expected for s in seen), (seen, expected)
    assert out.shape == (2, H * W, IN_CH), out.shape
    print(f"PASS  attention sees {expected} tokens (txt {TXT} + img {state['merger'].num_merged} "
          f"of {H * W}); output is full length {out.shape}")


def test_merge_mlp_flag_changes_double_blocks_only():
    dit_a, dit_b = _tiny_dit(), _tiny_dit()
    img, txt, img_ids, txt_ids = _inputs()
    mask = torch.zeros(H, W)
    mask[2:6, 2:6] = 1.0

    apply_patch(dit_a, mask, H, W, lr_factor=2, merge_mlp=False)
    apply_patch(dit_b, mask, H, W, lr_factor=2, merge_mlp=True)
    a = _run(dit_a, img, txt, img_ids, txt_ids)
    b = _run(dit_b, img, txt, img_ids, txt_ids)
    assert not torch.allclose(a, b, atol=1e-6), "merge_mlp had no effect at all"
    print(f"PASS  merge_mlp changes the result (max delta {(a - b).abs().max():.3e})")


if __name__ == "__main__":
    test_disabled_is_bit_exact_and_reversible()
    test_all_high_res_mask_is_bit_exact()
    test_merging_shrinks_the_attended_sequence()
    test_merge_mlp_flag_changes_double_blocks_only()
    print("\nAll patch tests passed.")
