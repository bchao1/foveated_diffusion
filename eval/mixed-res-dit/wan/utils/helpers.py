import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

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

def unified_scheduler_step(
    scheduler,
    t,
    latents_lr,
    latents_hr,
    noise_pred_lr,
    noise_pred_hr,
    ):
    # Update LR & HR latents together via concat on flattened spatial-temporal dims
    # Shapes: LR [B, D, F_lr, H_lr, W_lr], HR [B, D, F_hr, H_hr, W_hr]

    bsz, D_lt, f_lr, h_lr, w_lr   = latents_lr.shape
    _bsz, _D_lt, f_hr, h_hr, w_hr = latents_hr.shape 

    latents_lr_flat = latents_lr.reshape(bsz, D_lt, -1)
    latents_hr_flat = latents_hr.reshape(bsz, D_lt, -1)
    noise_lr_flat = noise_pred_lr.reshape(bsz, D_lt, -1)
    noise_hr_flat = noise_pred_hr.reshape(bsz, D_lt, -1)

    latents_cat = torch.cat([latents_lr_flat, latents_hr_flat], dim=2)
    noise_cat = torch.cat([noise_lr_flat, noise_hr_flat], dim=2)

    latents_cat, latents_cat_x0 = scheduler.step(noise_cat, t, latents_cat, return_dict=False)

    latents_lr_flat, latents_hr_flat = latents_cat.split(
        [latents_lr_flat.shape[2], latents_hr_flat.shape[2]], dim=2
    )
    latents_lr_x0_flat, latents_hr_x0_flat = latents_cat_x0.split(
        [latents_lr_flat.shape[2], latents_hr_flat.shape[2]], dim=2
    )

    latents_lr_new = latents_lr_flat.reshape(bsz, D_lt, f_lr, h_lr, w_lr)
    latents_hr_new = latents_hr_flat.reshape(bsz, D_lt, f_hr, h_hr, w_hr)
    latents_lr_x0 = latents_lr_x0_flat.reshape(bsz, D_lt, f_lr, h_lr, w_lr)
    latents_hr_x0 = latents_hr_x0_flat.reshape(bsz, D_lt, f_hr, h_hr, w_hr)

    return latents_lr_new, latents_hr_new, latents_lr_x0, latents_hr_x0


def feather_mask_from_bool(mask: torch.Tensor, radius=(1.0, 28.0, 28.0)) -> torch.Tensor:
    """
    Convert a hard mask into a soft feathered mask.

    Supports:
    - video mask: (B, 1, T, H, W), radius=(rt, rh, rw)
    - image mask: (B, 1, H, W), radius=(rh, rw) or scalar

    For anisotropic radius, distances are normalized per axis so each radius
    dimension controls transition width on its own axis.
    """
    if mask.ndim not in (4, 5) or mask.shape[1] != 1:
        raise ValueError(f"Expected mask shape (B,1,H,W) or (B,1,T,H,W), got {tuple(mask.shape)}")

    # Accept bool or {0,1} numeric mask.
    if mask.dtype == torch.bool:
        mask_bool = mask
    else:
        mask_bool = mask > 0.5

    if mask.ndim == 5:
        if isinstance(radius, (int, float)):
            rt = rh = rw = float(radius)
        else:
            if len(radius) != 3:
                raise ValueError(f"Video mask expects radius as (rt,rh,rw), got {radius}")
            rt, rh, rw = (float(radius[0]), float(radius[1]), float(radius[2]))
        rt, rh, rw = max(rt, 1e-6), max(rh, 1e-6), max(rw, 1e-6)
        sampling = (1.0 / rt, 1.0 / rh, 1.0 / rw)
    else:
        if isinstance(radius, (int, float)):
            rh = rw = float(radius)
        else:
            if len(radius) == 3:
                _, rh, rw = radius
            elif len(radius) == 2:
                rh, rw = radius
            else:
                raise ValueError(f"Image mask expects radius as scalar or (rh,rw), got {radius}")
            rh, rw = float(rh), float(rw)
        rh, rw = max(rh, 1e-6), max(rw, 1e-6)
        sampling = (1.0 / rh, 1.0 / rw)

    device = mask.device
    mask_np = mask_bool.detach().cpu().numpy()
    alpha_list = []

    for i in range(mask_np.shape[0]):
        mb = mask_np[i, 0].astype(np.uint8)
        di = distance_transform_edt(mb, sampling=sampling)
        do = distance_transform_edt(1 - mb, sampling=sampling)
        signed = di - do
        alpha = 0.5 * (1.0 + np.tanh(signed))
        alpha_list.append(alpha.astype(np.float32))

    alpha_np = np.stack(alpha_list, axis=0)
    alpha_t = torch.from_numpy(alpha_np).to(device=device, dtype=torch.float32).unsqueeze(1)
    return alpha_t

def transform_base_mask(
    base_mask: torch.Tensor,
    new_scale: int, 
    patch_size = None,
    ):
    """
    base_mask: low-res, patchified
    """
    if base_mask is None: return base_mask
    p_t, p_h, p_w = patch_size
    if new_scale > 1:
        mask_hr_latent = base_mask.repeat_interleave(new_scale, dim=2).repeat_interleave(new_scale, dim=3)
    else:
        mask_hr_latent = base_mask
    mask_hr_latent = (
        mask_hr_latent.repeat_interleave(p_t, dim=1)
        .repeat_interleave(p_h, dim=2)
        .repeat_interleave(p_w, dim=3)
    )
    mask_hr_latent = mask_hr_latent.unsqueeze(1)
    return mask_hr_latent