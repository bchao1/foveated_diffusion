import cv2
import numpy as np
import torch
from ultralytics.utils import ops

@torch.no_grad()
def refine_mask_w_seg(model, img_tensor, mask_tensor, target_k, imgsz=640, conf=0.25, iou=0.7, pool_threshold=0.05, overlap_threshold=0.1, verbose=False):
    """
    img_tensor: uint8, 0..255; (b, 3, h, w) = (1, 3, 512, 512) 
    mask_tensor: bool; (b, 1, lh, lw) = (1, 1, 32, 32)
    """
    b, _, h, w  = img_tensor.shape
    _, _, lh, lw = mask_tensor.shape
    img_np = img_tensor.detach().permute(0,2,3,1).cpu().numpy() # (b,h,w,3)
    mask_np = mask_tensor.detach().cpu().numpy()
    # RGB -> BGR, uint8, 0..255, H,W,3 or list[np.ndarray]
    img_np_list = [img_np[i, :, :, ::-1] for i in range(b)]
    results = model.predict(img_np_list, conf=conf, iou=iou, imgsz=imgsz, verbose=verbose)
    for i in range(b):
        mask_np[i, 0] = get_refined_saliency_mask(
            results[i],
            mask_np[i, 0], # np.bool (lh, lw)
            h, w, lh, lw,
            target_k,
            pool_threshold,
            overlap_threshold,
        )
    new_mask_tensor = torch.from_numpy(mask_np).to(mask_tensor.device) # (1, 1, 32, 32)
    return new_mask_tensor

def get_refined_saliency_mask(
    result,
    seed,
    h, w, lh, lw,
    target_k,
    pool_threshold=0.05,
    overlap_threshold=0.1,
) -> np.ndarray | None:
    """
    Returns:
        Binary mask as np.ndarray of bool
    """
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None

    ps_h, ps_w = h // lh, w // lw
    hh, ww = lh * ps_h, lw * ps_w

    # 1) seed
    # print(seed.shape, seed.dtype) # np.bool (lh, lw) = (32, 32)

    # 2) Downsample each instance mask to latent by max-pooling, then merge by label.
    masks = result.masks.data
    if masks.ndim == 2:
        masks = masks[None]
    if masks.shape[1:] != (h, w):
        masks = ops.scale_masks(masks[None].float(), (h, w))[0] > 0.5
    else:
        masks = masks > 0

    cls_ids = result.boxes.cls.to(device=masks.device, dtype=torch.int64)
    n = min(masks.shape[0], cls_ids.shape[0])
    if n == 0:
        return np.zeros((lh, lw), dtype=bool)

    masks_np = masks[:n, :hh, :ww].detach().cpu().numpy()
    cls_np = cls_ids[:n].detach().cpu().numpy().astype(np.int64)
    pooled = masks_np.reshape(n, lh, ps_h, lw, ps_w).sum(axis=(2, 4))  # (N, lh, lw)
    pooled = pooled > pool_threshold*ps_h*ps_w

    labels_present, inv = np.unique(cls_np, return_inverse=True)
    label_masks = np.zeros((len(labels_present), lh, lw), dtype=bool)
    np.logical_or.at(label_masks, inv, pooled)

    # De-overlap before ranking: if multiple labels cover the same latent pixel, keep only the smallest-area label.
    raw_area = label_masks.reshape(label_masks.shape[0], -1).sum(axis=1).astype(np.float32)
    if label_masks.shape[0] > 0:
        weighted = np.where(label_masks, raw_area[:, None, None], np.inf)
        winner = weighted.argmin(axis=0)  # (lh, lw)
        has_any = label_masks.any(axis=0)
        label_masks = has_any[None, :, :] & (winner[None, :, :] == np.arange(label_masks.shape[0])[:, None, None])

    area = label_masks.reshape(label_masks.shape[0], -1).sum(axis=1).astype(np.int32)
    valid = area > 0
    if not np.any(valid):
        return np.zeros((lh, lw), dtype=bool)
    labels_present = labels_present[valid]
    label_masks = label_masks[valid]
    area = area[valid]

    inter = (label_masks & seed[None, :, :]).reshape(label_masks.shape[0], -1).sum(axis=1).astype(np.float32)
    overlap_ratio = inter / np.maximum(area, 1)
    valid_overlap = overlap_ratio > overlap_threshold
    if not np.any(valid_overlap):
        return np.zeros((lh, lw), dtype=bool)
    labels_present = labels_present[valid_overlap]
    label_masks = label_masks[valid_overlap]
    area = area[valid_overlap]
    overlap_ratio = overlap_ratio[valid_overlap]
    order = np.lexsort((area, -overlap_ratio))

    # 3) Greedy add: skip labels that would exceed target; include at least one.
    selected = np.zeros(label_masks.shape[0], dtype=bool)
    current_area = 0
    for idx in order.tolist():
        next_area = current_area + int(area[idx])
        if next_area <= target_k:
            selected[idx] = True
            current_area = next_area
    if not selected.any():
        selected[order[0]] = True

    saliency = np.any(label_masks[selected], axis=0)

    # 4) Refine mask (holes + tiny isolated islands).
    return _refine_saliency_latent_mask(saliency, target_k)

def _refine_saliency_latent_mask(
    saliency: np.ndarray,
    target_k: int,
) -> np.ndarray:
    """
    Refine binary saliency mask
    1) opening  = remove tiny isolated islands
    2) closing  = fill small holes

    Args:
        saliency: input mask, bool / 0-1 / 0-255 all acceptable
    Returns:
        Binary mask as np.ndarray of bool
    """
    refined = saliency > 0
    # return refined

    # 1) remove tiny isolated islands:
    ncc, cc, stats, _ = cv2.connectedComponentsWithStats(refined.astype(np.uint8), connectivity=4)
    if ncc > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        min_island_area = max(2.0, float(target_k) * 0.02)
        keep = areas >= min_island_area
        keep_cc = np.zeros(ncc, dtype=bool)
        keep_cc[1:] = keep
        refined = keep_cc[cc]
        if not refined.any():
            refined = cc == (1 + int(np.argmax(areas)))

    # 2) fill small holes
    bg = (~refined).astype(np.uint8)
    ncc_bg, cc_bg, stats_bg, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)

    if ncc_bg > 1:
        h, w = refined.shape

        # Find all background connected components touching the image boundary:
        # these are external background, not holes.
        border_labels = np.unique(np.concatenate([
            cc_bg[0, :],        # top
            cc_bg[h - 1, :],    # bottom
            cc_bg[:, 0],        # left
            cc_bg[:, w - 1],    # right
        ]))

        is_hole = np.ones(ncc_bg, dtype=bool)
        is_hole[0] = False               # label 0 is not an actual component
        is_hole[border_labels] = False   # Boundary-touching components are external background

        # Hole area threshold
        hole_areas = stats_bg[:, cv2.CC_STAT_AREA].astype(np.float32)
        max_hole_area = max(2.0, float(target_k) * 0.01)

        fill_labels = np.where(is_hole & (hole_areas <= max_hole_area))[0]
        if len(fill_labels) > 0:
            refined[np.isin(cc_bg, fill_labels)] = True

    return refined.astype(bool)
