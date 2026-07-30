import torch
import torch.nn.functional as F

def binary_closing(mask: torch.Tensor, ksize: int = 3) -> torch.Tensor:
    """
    Apply morphological closing to a binary mask (dilation -> erosion),
    filling holes no larger than the kernel size.
    An odd ksize (3/5/7...) is recommended. This preserves shape as much
    as possible without globally expanding the final result.
    Supports (H, W) or (B, 1, H, W).
    """
    assert ksize % 2 == 1, "ksize must be odd"
    x = mask.clone().float()
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)

    pad = ksize // 2

    # Dilation: local maximum
    dilated = F.max_pool2d(x, kernel_size=ksize, stride=1, padding=pad)

    # Erosion: local minimum (negate so min becomes max)
    eroded = -F.max_pool2d(-dilated, kernel_size=ksize, stride=1, padding=pad)

    closed = (eroded > 0.5).to(mask.dtype)
    if mask.dim() == 2:
        return closed[0,0]
    elif mask.dim() == 3:
        return closed[:,0]
    return closed

def fill_mask_with_raft(
    video: torch.Tensor,
    latent_mask: torch.Tensor,
    raft_model,
    thresh: float = 0.5,   # After nearest-neighbor splatting, >0 counts as a hit; 0.5 also works
) -> torch.Tensor:

    if video.ndim != 5:
        raise ValueError(f"Expected video with 5 dims (B,3,T,H,W), got {video.shape}")
    if latent_mask.ndim != 4:
        raise ValueError(f"Expected latent_mask with 4 dims (B,T,H_l,W_l), got {latent_mask.shape}")

    video = video.float()
    latent_mask = latent_mask.to(torch.bool)

    b, _, t, h, w = video.shape
    mask_b, mask_t, h_l, w_l = latent_mask.shape
    if mask_b != b or mask_t != t:
        raise ValueError("Video and latent_mask batch/time dims mismatch.")
    if h % h_l != 0 or w % w_l != 0:
        raise ValueError("Video (H,W) must be divisible by latent (H_l,W_l).")
    if h % 8 != 0 or w % 8 != 0:
        raise ValueError("Video (H,W) must be divisible by 8.")
    if t < 2:
        return latent_mask.clone()

    scale_y = h // h_l
    scale_x = w // w_l

    # Downsample RAFT flow to latent resolution and normalize it into
    # (H_l, W_l) pixel units using the spatial scale factors.
    def _compute_flow_pairs(frames1: torch.Tensor, frames2: torch.Tensor) -> torch.Tensor:
        nonlocal h_l, w_l
        total_pairs = frames1.shape[0]
        with torch.no_grad():
            flow_full = raft_model(frames1, frames2)[-1]           # (total_pairs, 2, H, W)
        flow_full = flow_full.view(b, -1, 2, h, w)                 # (B, T-1, 2, H, W)
        flow_flat = flow_full.view(total_pairs, 2, h, w)
        flow_latent = F.interpolate(flow_flat, size=(h_l, w_l), mode="bilinear", align_corners=False)
        flow_latent = flow_latent.view(b, -1, 2, h_l, w_l)
        flow_latent[:, :, 0].div_(scale_x)  # dx on W_l
        flow_latent[:, :, 1].div_(scale_y)  # dy on H_l
        return flow_latent

    video_btchw = video.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, 3, H, W)
    frames_prev = video_btchw[:, :-1].reshape(-1, 3, h, w)
    frames_next = video_btchw[:,  1:].reshape(-1, 3, h, w)
    forward_flow  = _compute_flow_pairs(frames_prev, frames_next)  # t -> t+1
    backward_flow = _compute_flow_pairs(frames_next, frames_prev)  # t -> t-1

    # Precompute and reuse coordinate grids and batch offsets.
    device = video.device
    H, W = h_l, w_l
    dtype_coord = torch.float32  # Use float32 for coordinate math.

    ys_base, xs_base = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype_coord),
        torch.arange(W, device=device, dtype=dtype_coord),
        indexing="ij",
    )
    xsB = xs_base.unsqueeze(0).expand(b, -1, -1)  # (B,H,W)
    ysB = ys_base.unsqueeze(0).expand(b, -1, -1)  # (B,H,W)
    batch_offsets = (torch.arange(b, device=device) * (H * W)).view(b, 1, 1)  # (B,1,1)

    # Reusable output buffer to reduce repeated allocations; dtype matches mask_frame.
    def _make_out_buffer(example: torch.Tensor):
        return torch.zeros((b, H, W), device=example.device, dtype=example.dtype)

    # Nearest-neighbor forward splat: each source maps to a single pixel.
    def _splat_forward_nearest(mask_frame: torch.Tensor, flow_frame: torch.Tensor, out_buf: torch.Tensor) -> torch.Tensor:
        # mask_frame: (B,H,W) float/bool；flow_frame: (B,2,H,W) dx,dy
        out_buf.zero_()

        dx = flow_frame[:, 0].to(dtype_coord)
        dy = flow_frame[:, 1].to(dtype_coord)

        # Integer destination coordinates (nearest neighbor). Uses round; if
        # banker's rounding is a concern, switch to floor(x + 0.5) or a custom rule.
        x_r = torch.round(xsB + dx)
        y_r = torch.round(ysB + dy)

        valid = (x_r >= 0) & (x_r <= W - 1) & (y_r >= 0) & (y_r <= H - 1)
        x_r = x_r.clamp(0, W - 1).long()
        y_r = y_r.clamp(0, H - 1).long()

        flat_idx = (batch_offsets + y_r * W + x_r).reshape(-1)  # (B*H*W,)

        src_val = (mask_frame * valid.to(mask_frame.dtype)).reshape(-1)  # Invalid positions become 0.
        out_flat = out_buf.view(-1)
        out_flat.index_add_(0, flat_idx, src_val)

        return out_buf

    # Time propagation: forward first, then backward, using forward splatting in both directions.
    track_mask = latent_mask.clone()
    work_buf = _make_out_buffer(track_mask[:, 0].float())

    # Forward: t -> t+1, using forward_flow for forward splatting.
    mask_buffer = track_mask[:, 0].float()
    for idx in range(t - 1):
        splat = _splat_forward_nearest(mask_buffer, forward_flow[:, idx], work_buf)
        track_mask[:, idx + 1] |= (splat > thresh)
        mask_buffer = track_mask[:, idx + 1].float()

    # Backward: t -> t-1, using backward_flow to forward-splat into the previous frame.
    mask_buffer = track_mask[:, -1].float()
    for rev_idx in range(t - 1, 0, -1):
        splat = _splat_forward_nearest(mask_buffer, backward_flow[:, rev_idx - 1], work_buf)
        track_mask[:, rev_idx - 1] |= (splat > thresh)
        mask_buffer = track_mask[:, rev_idx - 1].float()
    
    track_mask = binary_closing(track_mask.view(mask_b*mask_t, 1, h_l, w_l)).view(mask_b, mask_t, h_l, w_l)

    return track_mask
