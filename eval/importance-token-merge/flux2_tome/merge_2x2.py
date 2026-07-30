"""
Mask-driven local block merging for FLUX.2.

This is ToMe's merge/unmerge contract (mean over each group, copy on unmerge) with the
grouping fixed by a foveation mask instead of chosen by feature similarity:

  - tokens inside a high-res block pass through untouched (each is its own group)
  - the lr_factor x lr_factor tokens of a low-res block form one group, represented by
    the block's top-left token

The block-level high/low decision replicates `_prepare_latent_image_ids_foveation` in
`flux2_image_foveated.py` exactly (a block is high-res iff *any* of its tokens is masked),
so the merged token count matches our foveated pipeline's token count for the same mask.

Nothing here imports or mutates DiffSynth.
"""

import torch


class BlockMerger:
    """Precomputed merge/unmerge for one (mask, resolution, lr_factor) triple.

    The grouping depends only on the mask, so it is built once and reused for every
    block and every denoising step.

    Attributes:
        keep_index: [M] ascending indices of the surviving tokens, in raster order.
        group_id:   [N] for each original token, its index in the merged sequence.
        num_merged: M, the merged image-token count.
    """

    def __init__(self, foveation_mask, height, width, lr_factor=2):
        """
        Args:
            foveation_mask: [height, width] tensor, 1 = high-res. Same object our
                pipeline receives from `create_foveation_mask`.
            height, width: latent token grid dims (image_size // 16).
            lr_factor: periphery downsample factor; 2 gives 2x2 blocks.
        """
        if height % lr_factor or width % lr_factor:
            raise ValueError(
                f"height ({height}) and width ({width}) must be divisible by lr_factor ({lr_factor})"
            )

        device = foveation_mask.device
        f = lr_factor
        h_d, w_d = height // f, width // f
        n_per_block = f * f
        num_tokens = height * width

        # Block is high-res iff any of its tokens is masked (matches the pipeline).
        mask_blocks = (
            foveation_mask.view(h_d, f, w_d, f).permute(0, 2, 1, 3).reshape(h_d, w_d, n_per_block)
        )
        is_high_res_block = mask_blocks.sum(dim=-1) > 0  # [h_d, w_d]

        # Per-token block membership, in raster order.
        rows = torch.arange(height, device=device).view(height, 1).expand(height, width)
        cols = torch.arange(width, device=device).view(1, width).expand(height, width)
        block_row, block_col = rows // f, cols // f
        token_is_hr = is_high_res_block[block_row, block_col]  # [height, width]

        # Representative token of each group: itself if high-res, else the block's top-left.
        top_left = (block_row * f) * width + (block_col * f)
        own = rows * width + cols
        representative = torch.where(token_is_hr, own, top_left).reshape(-1)  # [N]

        # A token survives iff it is its own representative.
        keep_mask = representative == torch.arange(num_tokens, device=device)
        keep_index = keep_mask.nonzero(as_tuple=True)[0]  # [M], ascending

        rank = torch.full((num_tokens,), -1, dtype=torch.long, device=device)
        rank[keep_index] = torch.arange(keep_index.numel(), device=device)
        group_id = rank[representative]  # [N]
        if bool((group_id < 0).any()):
            raise RuntimeError("token assigned to a dropped representative; grouping is inconsistent")

        counts = torch.bincount(group_id, minlength=keep_index.numel()).to(torch.float32)

        self.keep_index = keep_index
        self.group_id = group_id
        self.num_tokens = num_tokens
        self.num_merged = int(keep_index.numel())
        self._inv_counts = (1.0 / counts).view(1, -1, 1)

    def merge(self, x):
        """[B, N, C] -> [B, M, C], averaging each group (ToMe's `mode="mean"`)."""
        if x.shape[1] != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} tokens, got {x.shape[1]}")
        out = torch.zeros(
            x.shape[0], self.num_merged, x.shape[2], device=x.device, dtype=torch.float32
        )
        out.index_add_(1, self.group_id, x.to(torch.float32))
        return (out * self._inv_counts).to(x.dtype)

    def unmerge(self, x):
        """[B, M, C] -> [B, N, C], copying each group's value back (ToMe's unmerge)."""
        if x.shape[1] != self.num_merged:
            raise ValueError(f"expected {self.num_merged} tokens, got {x.shape[1]}")
        return x.index_select(1, self.group_id)

    def gather_positions(self, ids):
        """Subset [N, ...] position ids / rope tables to the surviving tokens.

        Merged groups inherit their top-left token's position, which is what
        `_prepare_latent_image_ids_foveation` assigns to a low-res block.
        """
        return ids.index_select(0, self.keep_index)
