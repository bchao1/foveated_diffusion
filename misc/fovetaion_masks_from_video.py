import argparse
import os
import sys
sys.path.append(".")

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter

from diffsynth.utils.fov_utils import create_foveation_mask_from_params, draw_foveation_circle_from_params
from diffsynth.utils.data import save_video
from diffsynth.core import UnifiedDataset

INFERENCE_SCALE = 0.5
TEMPORAL_SIGMA = 8.0
SPATIAL_SIGMA = 20.0
SPLINE_SMOOTHING = 0.5

def load_deepgaze_model(device):
    """Load DeepGaze IIE model with pretrained weights."""
    import deepgaze_pytorch
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
    model.eval()
    return model


def _frame_to_tensor(pil_img):
    """Convert a PIL Image to a float tensor [1, 3, H, W] in [0, 255]."""
    arr = np.array(pil_img).astype(np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _smooth_saliency_volume(smaps, temporal_sigma=TEMPORAL_SIGMA, spatial_sigma=SPATIAL_SIGMA):
    """
    Smooth saliency maps as a 3D (T, H, W) volume before parameter extraction.
    Heavy temporal smoothing (sigma=8 ~ averaging over 16 neighbors) makes the
    saliency nearly stationary; heavy spatial smoothing (sigma=20) consolidates
    scattered hotspots into a single coherent region.
    """
    volume = np.stack(smaps, axis=0)
    volume = gaussian_filter(volume, sigma=(temporal_sigma, spatial_sigma, spatial_sigma))
    smoothed = []
    for t in range(volume.shape[0]):
        s = volume[t]
        smax = s.max()
        if smax > 0:
            s = s / smax
        smoothed.append(s)
    return smoothed


def _saliency_to_params(smap, r_min=0.2, r_max=0.5, coverage=0.9, dense_threshold=0.3):
    """
    Extract (cx, cy, r) from a single (heavily smoothed) saliency heatmap
    using a simple weighted centroid over the dense core.

    After heavy 3D smoothing the map is typically unimodal, so no blob
    selection is needed.
    """
    h, w = smap.shape
    eps = 1e-8

    peak = smap.max()
    if peak < eps:
        return (0.0, 0.0), r_min

    thresh = dense_threshold * peak
    dense_mask = smap >= thresh
    dense_smap = smap * dense_mask
    dense_total = dense_smap.sum() + eps

    y_coords = np.arange(h, dtype=np.float64)
    x_coords = np.arange(w, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    cx_px = (dense_smap * x_grid).sum() / dense_total
    cy_px = (dense_smap * y_grid).sum() / dense_total

    cx_norm = (cx_px / w) - 0.5
    cy_norm = (cy_px / h) - 0.5

    dist = np.sqrt((x_grid - cx_px) ** 2 + (y_grid - cy_px) ** 2)
    dense_pixels = dense_mask.ravel()
    flat_dist = dist.ravel()[dense_pixels]
    flat_sal = dense_smap.ravel()[dense_pixels]

    sort_idx = np.argsort(flat_dist)
    cumsum = np.cumsum(flat_sal[sort_idx])
    target = coverage * dense_total

    hit = np.searchsorted(cumsum, target)
    hit = min(hit, len(flat_dist) - 1)
    radius_px = flat_dist[sort_idx[hit]]

    half_diag = 0.5 * np.sqrt(h**2 + w**2)
    r = radius_px / half_diag
    r = float(np.clip(r, r_min, r_max))

    return (float(cx_norm), float(cy_norm)), r


def _extract_all_params(smoothed_smaps, r_min=0.2, r_max=0.5):
    """Extract (cx, cy, r) at every frame from the heavily-smoothed saliency volume."""
    centers = []
    radii = []
    for smap in smoothed_smaps:
        center, r = _saliency_to_params(smap, r_min=r_min, r_max=r_max)
        centers.append(center)
        radii.append(r)
    return centers, radii


def _smooth_params(centers, radii, num_frames, r_min=0.2, r_max=0.5,
                   smoothing=SPLINE_SMOOTHING):
    """
    Fit a heavily-smoothed cubic spline through all per-frame parameters.

    The heavy 3D saliency smoothing already makes per-frame parameters
    fairly stable. The spline with s = num_frames * smoothing on top
    removes remaining jitter while faithfully following real movement
    trends (which show up consistently across many consecutive frames).
    """
    cx = np.array([c[0] for c in centers])
    cy = np.array([c[1] for c in centers])
    r = np.array(radii)
    t = np.arange(num_frames, dtype=np.float64)

    s = num_frames * smoothing
    if num_frames >= 4:
        cx_smooth = UnivariateSpline(t, cx, k=3, s=s)(t)
        cy_smooth = UnivariateSpline(t, cy, k=3, s=s)(t)
        r_smooth = UnivariateSpline(t, r, k=3, s=s)(t)
    elif num_frames >= 2:
        k = min(3, num_frames - 1)
        cx_smooth = UnivariateSpline(t, cx, k=k, s=0)(t)
        cy_smooth = UnivariateSpline(t, cy, k=k, s=0)(t)
        r_smooth = UnivariateSpline(t, r, k=k, s=0)(t)
    else:
        cx_smooth = cx.copy()
        cy_smooth = cy.copy()
        r_smooth = r.copy()

    r_smooth = np.clip(r_smooth, r_min, r_max)
    return list(zip(cx_smooth.tolist(), cy_smooth.tolist())), r_smooth.tolist()


def predict_saliency_maps(model, frames, device):
    """
    Full pipeline: half-res inference -> heavy 3D saliency smoothing ->
    per-frame parameter extraction -> heavily-smoothed cubic spline ->
    circular foveation masks.

    Returns:
        masks: (1, 1, T, H, W) foveation mask tensor.
        smoothed_smaps: list of (h_inf, w_inf) numpy arrays for debug viz.
        centers: list of (cx, cy) tuples (smoothed, one per frame).
        radii: list of float (smoothed, one per frame).
    """
    num_frames = len(frames)
    w_orig, h_orig = frames[0].size

    w_inf = int(w_orig * INFERENCE_SCALE)
    h_inf = int(h_orig * INFERENCE_SCALE)
    resized_frames = [f.resize((w_inf, h_inf), Image.LANCZOS) for f in frames]

    centerbias = torch.zeros(1, h_inf, w_inf).to(device)

    raw_smaps = []
    for i, frame in enumerate(resized_frames):
        img_tensor = _frame_to_tensor(frame).to(device)
        with torch.no_grad():
            log_density = model(img_tensor, centerbias)
        smap = np.exp(log_density[0, 0].cpu().numpy())
        smax = smap.max()
        if smax > 0:
            smap = smap / smax
        raw_smaps.append(smap)
        if (i + 1) % 10 == 0 or i == num_frames - 1:
            print(f"    Saliency: {i + 1}/{num_frames} frames")

    smoothed_smaps = _smooth_saliency_volume(raw_smaps)

    centers, radii = _extract_all_params(smoothed_smaps)

    centers, radii = _smooth_params(centers, radii, num_frames)

    masks = create_foveation_mask_from_params(
        keyframe_centers=centers,
        keyframe_radii=radii,
        num_frames=num_frames,
        height=h_orig,
        width=w_orig,
        sigma=10.0,
        device=torch.device(device),
        dtype=torch.float32,
    )

    return masks, smoothed_smaps, centers, radii


def main(args):
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=1,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading DeepGaze IIE saliency model...")
    model = load_deepgaze_model(device)

    print(f"Processing {len(dataset)} videos from dataset...")
    for idx in range(len(dataset)):
        data = dataset[idx]
        frames = data.get("video")
        if frames is None or not isinstance(frames, list) or len(frames) == 0:
            print(f"[{idx}] Skipping entry with no loaded video frames")
            continue

        raw_meta = dataset.data[idx % len(dataset.data)]
        video_name = os.path.splitext(os.path.basename(str(raw_meta.get("video", idx))))[0]

        print(f"[{idx}/{len(dataset)}] Processing: {video_name} ({len(frames)} frames, {frames[0].size[0]}x{frames[0].size[1]})")
        try:
            masks, smoothed_saliency, centers, radii = predict_saliency_maps(model, frames, device)

            fov_mask = masks.to(device)

            video = torch.stack([
                torch.from_numpy(np.array(f)).permute(2, 0, 1).float() / 255.0
                for f in frames
            ])
            video = video.permute(1, 0, 2, 3).unsqueeze(0).to(device)

            video_lr = F.interpolate(video, scale_factor=(1, 0.5, 0.5), mode='trilinear', align_corners=False)
            video_lr_up = F.interpolate(video_lr, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
            fov_video = fov_mask * video + (1 - fov_mask) * video_lr_up

            mask_path = os.path.join(args.output_dir, f"{video_name}_foveation_mask.mp4")
            saliency_path = os.path.join(args.output_dir, f"{video_name}_saliency.mp4")
            orig_path = os.path.join(args.output_dir, f"{video_name}_original.mp4")
            fov_path = os.path.join(args.output_dir, f"{video_name}_foveated.mp4")

            save_video((masks[0, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), mask_path, fps=15, quality=5)

            w_orig, h_orig = frames[0].size
            saliency_frames = []
            for smap in smoothed_saliency:
                s = torch.from_numpy(smap).float().unsqueeze(0).unsqueeze(0)
                s = F.interpolate(s, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                saliency_frames.append((s.squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8))
            save_video(saliency_frames, saliency_path, fps=15, quality=5)

            save_video(frames, orig_path, fps=15, quality=5)
            fov_frames = (fov_video[0].permute(1, 2, 3, 0).clamp(0, 1) * 255).byte().cpu().numpy()
            fov_frames_with_circle = draw_foveation_circle_from_params(
                list(fov_frames), centers, radii)
            save_video(fov_frames_with_circle, fov_path, fps=15, quality=5)

            print(f"  Saved: {mask_path}, {saliency_path}, {orig_path}, {fov_path}")
        except Exception as e:
            import traceback
            print(f"  Error processing {video_name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate foveation masks for videos using DeepGaze IIE saliency prediction")
    parser.add_argument("--dataset_base_path", type=str, default="/miele/liory/datasets/vchitect/videos")
    parser.add_argument("--dataset_metadata_path", type=str, default="/miele/liory/datasets/vchitect/metadata.csv")
    parser.add_argument("--data_file_keys", type=str, default="image,video")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--output_dir", type=str, default="./foveation_masks")
    args = parser.parse_args()
    main(args)
