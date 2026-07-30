import contextlib
import os
from urllib.request import urlretrieve
from typing import Iterable

import torch
import torch.nn.functional as F

from sam2.build_sam import build_sam2_video_predictor
import sam2.sam2_video_predictor as sam2_video_predictor_module
import sam2.utils.misc as sam2_misc_module


def _to_obj_id_list(obj_ids: Iterable) -> list[int]:
    if isinstance(obj_ids, torch.Tensor):
        return [int(x) for x in obj_ids.detach().cpu().tolist()]
    return [int(x) for x in obj_ids]


def _pick_obj_mask(mask_logits, obj_ids, target_obj_id: int) -> torch.Tensor:
    ids = _to_obj_id_list(obj_ids)
    assert target_obj_id in ids, f"target_obj_id={target_obj_id} not found in obj_ids={ids}"

    idx = ids.index(target_obj_id)
    obj_mask = mask_logits[idx]
    if not isinstance(obj_mask, torch.Tensor):
        obj_mask = torch.as_tensor(obj_mask)
    obj_mask = obj_mask.squeeze()
    assert obj_mask.ndim == 2, f"unexpected SAM output shape after squeeze: {tuple(obj_mask.shape)}"
    return obj_mask > 0


_orig_sam2_load_video_frames = sam2_video_predictor_module.load_video_frames


def _normalize_video_tensor_input(video_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert input tensor to (T, 3, H, W).
    Supported layouts: (3, T, H, W), (T, 3, H, W), (T, H, W, 3).
    """
    if video_tensor.ndim != 4:
        raise ValueError(f"video tensor must be 4D, got {tuple(video_tensor.shape)}")

    if video_tensor.shape[0] == 3:
        return video_tensor.permute(1, 0, 2, 3).contiguous()
    if video_tensor.shape[1] == 3:
        return video_tensor.contiguous()
    if video_tensor.shape[-1] == 3:
        return video_tensor.permute(0, 3, 1, 2).contiguous()

    raise ValueError(
        f"unsupported video tensor layout: {tuple(video_tensor.shape)}; expected CTHW/TCHW/THWC with 3 channels"
    )


def _load_video_frames_from_tensor(
    video_tensor: torch.Tensor,
    image_size: int,
    offload_video_to_cpu: bool,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=torch.device("cuda"),
):
    images = _normalize_video_tensor_input(video_tensor)
    video_height, video_width = int(images.shape[-2]), int(images.shape[-1])

    images = images.float()
    if images.numel() > 0 and images.max() > 1.0:
        images = images / 255.0

    if images.shape[-2:] != (image_size, image_size):
        images = F.interpolate(
            images,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

    target_device = torch.device("cpu") if offload_video_to_cpu else compute_device
    images = images.to(target_device, non_blocking=True)
    img_mean_t = torch.tensor(img_mean, dtype=torch.float32, device=target_device)[:, None, None]
    img_std_t = torch.tensor(img_std, dtype=torch.float32, device=target_device)[:, None, None]
    images = (images - img_mean_t) / img_std_t
    return images, video_height, video_width


def _patched_load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
):
    if torch.is_tensor(video_path):
        return _load_video_frames_from_tensor(
            video_tensor=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            compute_device=compute_device,
        )
    return _orig_sam2_load_video_frames(
        video_path=video_path,
        image_size=image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        img_mean=img_mean,
        img_std=img_std,
        async_loading_frames=async_loading_frames,
        compute_device=compute_device,
    )


@torch.no_grad()
def prepare_sam2_tiny_video_predictor(ckpt_dir, device: str = "cuda:0"):
    """Prepare SAM 2.1 tiny predictor and checkpoint before inference."""
    # Disable tqdm progress bars used inside sam2_video_predictor.propagate_in_video.
    sam2_video_predictor_module.tqdm = lambda iterable, **kwargs: iterable
    # Allow predictor.init_state(...) to accept torch tensor videos directly.
    sam2_video_predictor_module.load_video_frames = _patched_load_video_frames
    sam2_misc_module.load_video_frames = _patched_load_video_frames

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "sam2.1_hiera_tiny.pt")
    if not os.path.exists(ckpt_path):
        urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
            ckpt_path,
        )

    return build_sam2_video_predictor(
        config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
        ckpt_path=ckpt_path,
        device=device,
        vos_optimized=False,
        apply_postprocessing=False,
    )


@torch.no_grad()
def propagate_mask_w_sam2_tiny(
    predictor,
    video_tensor_bcthw: torch.Tensor,
    base_mask_bhw: torch.Tensor,
    base_idx: int,
    device: str = "cuda:0",
    obj_id: int = 1,
    autocast_dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Propagate a base-frame mask to all frames with a prepared SAM predictor.
    """
    assert video_tensor_bcthw.ndim == 5 and video_tensor_bcthw.shape[1] == 3, f"video_tensor_bcthw must be (B,3,T,H,W), got {tuple(video_tensor_bcthw.shape)}"
    assert video_tensor_bcthw.dtype == torch.uint8, f"video_tensor_bcthw must be uint8, got {video_tensor_bcthw.dtype}"
    b, _, t, h, w = video_tensor_bcthw.shape

    assert isinstance(base_idx, int), f"base_idx must be int, got {type(base_idx)}"
    assert 0 <= base_idx < t, f"base_idx out of range: {base_idx}, T={t}"

    assert base_mask_bhw.ndim == 3, f"base_mask_bhw must be (B,H,W), got {tuple(base_mask_bhw.shape)}"
    assert base_mask_bhw.shape[0] == b, f"mask batch mismatch: B(video)={b} vs B(mask)={base_mask_bhw.shape[0]}"
    assert base_mask_bhw.shape[-2:] == (h, w), f"mask size mismatch: mask={tuple(base_mask_bhw.shape[-2:])} vs video={(h, w)}"
    base_mask_bhw = base_mask_bhw.to(torch.bool).detach()
    video_mask = torch.zeros((b, t, h, w), dtype=torch.bool, device=device)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast_dtype):
        for bi in range(b):
            state = predictor.init_state(video_tensor_bcthw[bi].contiguous()) # (3, T, H, W)
            predictor.reset_state(state)

            frame_idx, obj_ids, mask_logits = predictor.add_new_mask(
                inference_state=state,
                frame_idx=base_idx,
                obj_id=obj_id,
                mask=base_mask_bhw[bi], # (H, W)
            )
            video_mask[bi, int(frame_idx)] = _pick_obj_mask(mask_logits, obj_ids, target_obj_id=obj_id).to(device=device)

            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                state,
                start_frame_idx=base_idx,
                reverse=False,
            ):
                video_mask[bi, int(frame_idx)] = _pick_obj_mask(mask_logits, obj_ids, target_obj_id=obj_id).to(device=device)
            if base_idx > 0:
                for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                    state,
                    start_frame_idx=base_idx,
                    reverse=True,
                ):
                    video_mask[bi, int(frame_idx)] = _pick_obj_mask(mask_logits, obj_ids, target_obj_id=obj_id).to(device=device)

    return video_mask
