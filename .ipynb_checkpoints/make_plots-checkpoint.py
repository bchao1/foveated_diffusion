
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def extract_crops(img_path, output_folder):
    # 1. Setup
    img = imageio.imread(img_path)
    H, W = img.shape[:2]
    os.makedirs(output_folder, exist_ok=True)

    crop_size = 200
    shift_amount = 250
    
    # Colors
    color_center = '#FF00FF' # Magenta
    color_left = '#00FFFF'   # Cyan

    # 2. Calculate Coordinates
    # --- Center ---
    cy, cx = H // 2, W // 2
    y1_c = max(0, cy - (crop_size // 2))
    y2_c = min(H, cy + (crop_size // 2))
    x1_c = max(0, cx - (crop_size // 2))
    x2_c = min(W, cx + (crop_size // 2))

    # --- Shifted Left ---
    sx = cx - shift_amount
    y1_s = max(0, cy - (crop_size // 2))
    y2_s = min(H, cy + (crop_size // 2))
    x1_s = max(0, sx - (crop_size // 2))
    x2_s = min(W, sx + (crop_size // 2))
    
    # 3. Extract Data
    crop_center_data = img[y1_c:y2_c, x1_c:x2_c]
    
    crop_left_data = None
    if x2_s > x1_s and y2_s > y1_s:
        crop_left_data = img[y1_s:y2_s, x1_s:x2_s]

    # ==========================================
    # Helper Function to Save Crop with Border
    # ==========================================
    def save_crop_with_border(data, color, filename, dashed=False):
        # Create figure with no frame around the canvas, just the image
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(data)
        
        # Turn ON the axis so we can use the spines as borders
        ax.axis('on')
        
        # Remove ticks and labels (we only want the frame)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Color the spines (the 4 borders)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(5) # Very thick to be visible
            if dashed:
                spine.set_linestyle('--')

        # Save
        # bbox_inches='tight' usually cuts off half the spine thickness
        # pad_inches=0.02 gives just enough room for the full border width to show
        save_path = os.path.join(output_folder, filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=200)
        plt.close(fig)

    # 4. Save Crops
    save_crop_with_border(crop_center_data, color_center, 'crop_center_border.png')
    
    if crop_left_data is not None:
        save_crop_with_border(crop_left_data, color_left, 'crop_left_border.png', dashed=True)

    # ==========================================
    # 5. Save Original with Bounding Boxes
    # ==========================================
    fig_main, ax_main = plt.subplots(figsize=(10, 10))
    ax_main.imshow(img)
    
    # Box 1: Center
    rect_c = patches.Rectangle(
        (x1_c, y1_c), x2_c - x1_c, y2_c - y1_c,
        linewidth=3, edgecolor=color_center, facecolor='none', label='Center Crop'
    )
    ax_main.add_patch(rect_c)

    # Box 2: Left
    rect_l = patches.Rectangle(
        (x1_s, y1_s), x2_s - x1_s, y2_s - y1_s,
        linewidth=3, edgecolor=color_left, facecolor='none', linestyle='--', label='Left Crop'
    )
    ax_main.add_patch(rect_l)
    
    ax_main.legend(handles=[rect_c, rect_l], loc='upper right', framealpha=1, facecolor='white')
    ax_main.set_axis_off()
    
    plt.savefig(os.path.join(output_folder, 'original_with_bboxes.png'), 
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig_main)
    
    print(f"Processed {img_path}. Images saved to {output_folder}")

if __name__ == "__main__":
    extract_crops("results/0120_fix_packing_training_free_start_step_10_pack_fix/img_02.png", "paper_figures/mixed_res")
    extract_crops("results/0120_fix_packing_training_free_start_step_10_pack_fix/img_09.png", "paper_figures/high_res")
    
    
    
    