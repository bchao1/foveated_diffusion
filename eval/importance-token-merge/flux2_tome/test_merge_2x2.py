"""Correctness gate for BlockMerger. Runs standalone (CPU, no diffsynth needed).

    python -m flux2_tome.test_merge_2x2

The last test cross-checks the merged token count against our foveated pipeline's own
`_prepare_latent_image_ids_foveation`, and is skipped if diffsynth is not importable.
"""

import torch

from .merge_2x2 import BlockMerger

H = W = 64  # 1024px at /16
F = 2


def _rand_tokens(n, c=8, seed=0):
    return torch.randn(2, n, c, generator=torch.Generator().manual_seed(seed))


def test_all_high_res_is_identity():
    mask = torch.ones(H, W)
    m = BlockMerger(mask, H, W, lr_factor=F)
    assert m.num_merged == H * W, m.num_merged
    assert torch.equal(m.keep_index, torch.arange(H * W))
    x = _rand_tokens(H * W)
    assert torch.allclose(m.merge(x), x)
    assert torch.allclose(m.unmerge(m.merge(x)), x)
    print("PASS  all-high-res is identity")


def test_all_low_res_is_average_pool():
    mask = torch.zeros(H, W)
    m = BlockMerger(mask, H, W, lr_factor=F)
    assert m.num_merged == (H // F) * (W // F), m.num_merged

    x = _rand_tokens(H * W, seed=1)
    merged = m.merge(x)
    # Reference: 2x2 average pool over the raster grid.
    spatial = x.view(2, H, W, -1).permute(0, 3, 1, 2)
    ref = torch.nn.functional.avg_pool2d(spatial, F).permute(0, 2, 3, 1).reshape(2, -1, x.shape[-1])
    assert torch.allclose(merged, ref, atol=1e-5), (merged - ref).abs().max()

    # Unmerge copies the block mean back to all F*F positions.
    back = m.unmerge(merged).view(2, H // F, F, W // F, F, -1)
    assert torch.allclose(back, back[:, :, :1, :, :1, :].expand_as(back), atol=1e-5)
    print("PASS  all-low-res is 2x2 average pool + copy unmerge")


def test_mixed_mask_counts_and_fovea_untouched():
    mask = torch.zeros(H, W)
    mask[20:44, 20:44] = 1.0  # centered square, block-aligned
    m = BlockMerger(mask, H, W, lr_factor=F)

    blocks = mask.view(H // F, F, W // F, F).permute(0, 2, 1, 3).reshape(H // F, W // F, F * F)
    n_hr = int((blocks.sum(-1) > 0).sum())
    n_lr = (H // F) * (W // F) - n_hr
    assert m.num_merged == n_hr * F * F + n_lr, (m.num_merged, n_hr, n_lr)

    # Every token inside a high-res block must survive unchanged.
    x = _rand_tokens(H * W, seed=2)
    out = m.unmerge(m.merge(x)).view(2, H, W, -1)
    hr_tokens = (blocks.sum(-1) > 0).repeat_interleave(F, 0).repeat_interleave(F, 1)
    assert torch.allclose(out[:, hr_tokens], x.view(2, H, W, -1)[:, hr_tokens], atol=1e-5)
    print(f"PASS  mixed mask: {m.num_tokens} -> {m.num_merged} tokens, fovea bit-exact")


def test_unaligned_mask_matches_pipeline_token_count():
    """A block counts as high-res if *any* of its tokens is masked, so a mask whose
    boundary cuts through blocks must still agree with the pipeline."""
    try:
        from diffsynth.pipelines.flux2_image_foveated import Flux2FoveatedImagePipeline
    except Exception as e:  # noqa: BLE001
        print(f"SKIP  pipeline cross-check (diffsynth unavailable: {type(e).__name__})")
        return

    torch.manual_seed(3)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    mask = (((yy - 31.5) ** 2 + (xx - 27.5) ** 2).sqrt() < 13.3).float()  # deliberately unaligned

    m = BlockMerger(mask, H, W, lr_factor=F)
    ids, _, _ = Flux2FoveatedImagePipeline._prepare_latent_image_ids_foveation(
        1, H, W, torch.device("cpu"), torch.float32, mask, lr_factor=F
    )
    assert m.num_merged == ids.shape[0], (m.num_merged, ids.shape[0])

    # Position sets must match too (order differs: ours is raster, theirs is block-major).
    ours = m.gather_positions(torch.cartesian_prod(
        torch.zeros(1), torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32), torch.zeros(1),
    ))
    a = sorted(map(tuple, ours.tolist()))
    b = sorted(map(tuple, ids.tolist()))
    assert a == b, "position id sets differ from the foveated pipeline"
    print(f"PASS  unaligned mask matches pipeline exactly ({m.num_merged} tokens)")


if __name__ == "__main__":
    test_all_high_res_is_identity()
    test_all_low_res_is_average_pool()
    test_mixed_mask_counts_and_fovea_untouched()
    test_unaligned_mask_matches_pipeline_token_count()
    print("\nAll merge tests passed.")
