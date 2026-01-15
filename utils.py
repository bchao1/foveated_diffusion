import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio
import os
import cv2

def save_image(tensor, path, debug_visualization=False):
    # save tensor of shape [3, H, W] or [H, W] as image
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8) # [3, H, W] or [H, W]

    if tensor.ndim == 2:
        tensor = np.stack([tensor] * 3, axis=0)
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = np.concatenate([tensor] * 3, axis=0)

    if tensor.shape[0] == 3:
        img = tensor.transpose(1, 2, 0).copy()
    else:
        img = tensor

    if debug_visualization:
        scale = 10
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        img[::scale, :, 0] = 255
        img[::scale, :, 1] = 0
        img[::scale, :, 2] = 0
        img[:, ::scale, 0] = 255
        img[:, ::scale, 1] = 0
        img[:, ::scale, 2] = 0

    imageio.imwrite(path, img)

def create_foveation_mask(
    h: int, 
    w: int, 
    center: tuple[float, float] = (0.0, 0.0), 
    r: float = 0.25, 
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Creates a binary circular mask using PyTorch.

    Args:
        h, w (int): Dimensions of the mask.
        center (tuple): (x, y) coordinates (-0.5 to 0.5).
        r (float): Radius relative to half the diagonal.
        device: The device to create the tensor on (cpu or cuda).

    Returns:
        torch.Tensor: Boolean mask of shape (h, w).
    """
    # 1. Convert normalized center to pixel coordinates
    cx = (center[0] + 0.5) * w
    cy = (center[1] + 0.5) * h

    # 2. Convert normalized radius to pixel radius
    diagonal = (h**2 + w**2)**0.5
    radius_px = r * (diagonal / 2.0)

    # 3. Create coordinate grid
    # indexing='ij' ensures dimension 0 is H (y) and dimension 1 is W (x)
    y_range = torch.arange(h, device=device, dtype=torch.float32)
    x_range = torch.arange(w, device=device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')

    # 4. Calculate squared Euclidean distance
    dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2

    # 5. Create binary mask
    mask = dist_sq <= radius_px**2

    return mask

def visualize_2d_map(vector_map, title="2D Map"):
    """
    Visualizes a 2D vector map (H, W, 2) or (H, W, 3) using color coding.
    
    Args:
        vector_map: numpy array or torch tensor of shape (H, W, C). 
                    If C=3, it assumes index 0 is empty/latent and uses indices 1,2 as (y,x).
                    If C=2, it uses indices 0,1 as (y,x) or (x,y).
    """
    if torch.is_tensor(vector_map):
        vector_map = vector_map.detach().cpu().numpy()
        
    # Handle the specific 3-channel structure of latent_image_ids (0, y, x)
    if vector_map.shape[-1] == 3:
        # Extract only the Y, X components (indices 1 and 2)
        # Note: We treat index 2 as X (real part) and index 1 as Y (imaginary part) for visualization
        u = vector_map[..., 2] # x component
        v = vector_map[..., 1] # y component
    else:
        u = vector_map[..., 1] 
        v = vector_map[..., 0]

    # Convert to polar coordinates
    mag = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)

    # Normalize magnitude for visualization (0 to 1)
    if mag.max() > 0:
        mag_norm = mag / mag.max()
    else:
        mag_norm = mag

    # Map Angle (-pi to pi) to Hue (0 to 1)
    hue = (angle + np.pi) / (2 * np.pi)
    
    # Create HSV image
    # Hue: Direction, Saturation: 1.0, Value: Magnitude
    hsv = np.zeros((vector_map.shape[0], vector_map.shape[1], 3))
    hsv[..., 0] = hue
    hsv[..., 1] = 1.0 # Keep saturation full for clarity
    hsv[..., 2] = mag_norm 

    # Convert HSV to RGB
    rgb = mcolors.hsv_to_rgb(hsv)

    # Add a color wheel legend for reference
    #  - concept implied here
    # return image 
    return rgb # [H, W, 3]

def create_warp_mask(H, W, device='cpu'):
    # Create a grid of coordinates
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32), 
        torch.arange(W, device=device, dtype=torch.float32), 
        indexing='ij'
    )

    # --- 1. Y Warp (Vertical) ---
    # Shift columns up/down based on their X position
    # Amplitude: 10% of total height
    amp_y = 0.1 * H
    # Frequency: scales with W so we get ~2 waves across the width
    freq_x = (4.0 * 3.14159) / W 
    flow_y = amp_y * torch.sin(x * freq_x)

    # --- 2. X Warp (Horizontal) ---
    # Shift rows left/right based on their Y position
    # Amplitude: 10% of total width
    amp_x = 0.1 * W 
    # Frequency: scales with H so we get ~2 waves down the height
    freq_y = (4.0 * 3.14159) / H 
    flow_x = amp_x * torch.sin(y * freq_y)

    # Stack them: (H, W, 2) -> (dY, dX)
    flow_field = torch.stack([flow_y, flow_x], dim=-1)
    
    return flow_field

def make_videos(folder, output_file, fps=5):
    image_files = [f for f in sorted(os.listdir(folder)) if f.startswith("img") and f.endswith('.png')]
    mask_files = [f for f in sorted(os.listdir(folder)) if f.startswith("foveation_mask_high_res") and f.endswith('.png')]
    image_files.sort()
    mask_files.sort()
    print(len(image_files), len(mask_files))
    images = [imageio.imread(os.path.join(folder, f)) for f in image_files]
    masks = [imageio.imread(os.path.join(folder, f)) for f in mask_files]
    with imageio.get_writer(output_file, fps=fps) as writer:
        for image, mask in zip(images, masks):
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            combined = np.concatenate([image, mask], axis=1)
            writer.append_data(combined)
    writer.close()