import torch
import numpy as np
import cv2

def quick_benchmark_gpu(model, *args, iters=10, warmup=5, name='bench', **kwargs):
    for _ in range(warmup): model(*args, **kwargs)
    times_ms = []
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        starter.record()
        model(*args, **kwargs)
        ender.record()
        torch.cuda.synchronize()
        times_ms.append(starter.elapsed_time(ender)) 
    print(f'{name}:', sum(times_ms)/len(times_ms), 'ms')

def feather_mask_from_bool(mask: torch.Tensor, radius: float = 28) -> torch.Tensor:
    """
    Convert a hard boolean mask (B, 1, H, W) into a soft feathered mask
    with the same shape, similar to the alpha produced by distance_feather().

    Args:
        mask: torch.Tensor of shape (B, 1, H, W), dtype=bool
        radius: feather radius controlling transition width

    Returns:
        alpha: torch.Tensor of shape (B, 1, H, W), dtype=float32,
               values in [0, 1]
    """
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask shape (B, 1, H, W), got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise TypeError(f"Expected bool mask, got {mask.dtype}")

    radius = max(float(radius), 1e-6)

    device = mask.device
    b, _, h, w = mask.shape

    alpha_list = []

    # OpenCV works on CPU numpy arrays, so move temporarily to CPU
    mask_np = mask.detach().cpu().numpy()

    for i in range(b):
        # shape: (H, W), values {0,1}
        mb = mask_np[i, 0].astype(np.uint8)

        # Distance inside the mask
        di = cv2.distanceTransform(mb, cv2.DIST_L2, 5)

        # Distance outside the mask
        do = cv2.distanceTransform(1 - mb, cv2.DIST_L2, 5)

        signed = di - do
        alpha = 0.5 * (1.0 + np.tanh(signed / radius))

        alpha_list.append(alpha)

    alpha_np = np.stack(alpha_list, axis=0)  # (B, H, W)
    alpha_t = torch.from_numpy(alpha_np).to(device=device, dtype=torch.float32)
    alpha_t = alpha_t.unsqueeze(1)  # (B, 1, H, W)

    return alpha_t

def align_latent_mean_shift_flexible(
    ref_latents_x0: torch.Tensor,
    new_latents_x0: torch.Tensor,
):
    """
    Cupports:
      - token latents: [B, N, C]
      - unpacked latents: [B, C, H, W]
    If inputs are 4D, only batch/channel must match; H/W can differ.
    Outputs are converted back to 4D using `new_latents_x0` spatial size.
    """

    if ref_latents_x0.ndim not in (3, 4) or new_latents_x0.ndim not in (3, 4):
        raise ValueError(
            f"Expected 3D/4D latents, got ref={ref_latents_x0.shape}, new={new_latents_x0.shape}."
        )
    if ref_latents_x0.ndim != new_latents_x0.ndim:
        raise ValueError(
            f"Latent rank mismatch: ref={ref_latents_x0.shape}, new={new_latents_x0.shape}."
        )

    input_is_4d = ref_latents_x0.ndim == 4
    ref_hw = None
    new_hw = None

    if input_is_4d:
        b_ref, c_ref, h_ref, w_ref = ref_latents_x0.shape
        b_new, c_new, h_new, w_new = new_latents_x0.shape
        if b_ref != b_new or c_ref != c_new:
            raise ValueError(
                f"4D batch/channel mismatch between ref={ref_latents_x0.shape} and new={new_latents_x0.shape}."
            )
        ref_hw = (h_ref, w_ref)
        new_hw = (h_new, w_new)
        ref_tokens = ref_latents_x0.permute(0, 2, 3, 1).reshape(b_ref, h_ref * w_ref, c_ref)
        new_tokens = new_latents_x0.permute(0, 2, 3, 1).reshape(b_new, h_new * w_new, c_new)
    else:
        ref_tokens = ref_latents_x0
        new_tokens = new_latents_x0
        if ref_tokens.shape[0] != new_tokens.shape[0] or ref_tokens.shape[2] != new_tokens.shape[2]:
            raise ValueError(
                f"Batch/channel mismatch between ref={ref_tokens.shape} and new={new_tokens.shape}."
            )

    def _get_mean(x: torch.Tensor):
        return x.mean(dim=1, keepdim=True)

    ref_mean = _get_mean(ref_tokens).to(new_tokens.dtype)
    new_mean = _get_mean(new_tokens)
    mean_shift = ref_mean - new_mean

    aligned_tokens = new_tokens + mean_shift

    if not input_is_4d:
        return aligned_tokens, mean_shift

    b, _, c = aligned_tokens.shape
    h, w = new_hw
    aligned_4d = aligned_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    mean_shift_for_return = mean_shift
    if mean_shift_for_return.shape[-1] == 1 and c != 1:
        mean_shift_for_return = mean_shift_for_return.expand(-1, -1, c)
    mean_shift_4d = mean_shift_for_return.reshape(b, 1, c).permute(0, 2, 1).reshape(b, c, 1, 1).contiguous()
    return aligned_4d, mean_shift_4d

def transform_base_mask(
    base_mask: torch.Tensor,
    new_scale: int, 
    patch_size = None,
    ):
    """
    base_mask: low-res, patchified
    """
    assert base_mask.ndim == 4
    if new_scale > 1:
        mask = base_mask.repeat_interleave(new_scale, dim=2).repeat_interleave(new_scale, dim=3)
    else:
        mask = base_mask
    if patch_size > 1:
        mask = mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    return mask
