import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..") # Uncomment if needed
from utils import save_image # Uncomment if needed
from PIL import Image
import numpy as np


def read_image(path, size=(128, 128)):
    # Standard helper to read and prep image
    img = Image.open(path)
    img = img.resize(size)
    img = np.array(img)
    img = img.transpose(2, 0, 1)
    img = img / 255.0
    img = torch.from_numpy(img).float()
    return img.unsqueeze(0)

class TokenDownsampler(nn.Module):
    def __init__(self, num_layers=2, hidden_dim=64, bias=False, debug=False):
        super().__init__()
        self.debug = debug
        
        if self.debug:
            # Debug: Pick top-left pixel (Nearest Neighbor equivalent for Downsample)
            self.mlp = nn.Linear(4, 1, bias=False)
            with torch.no_grad():
                self.mlp.weight.fill_(0)
                self.mlp.weight[0, 0] = 1.0
        else:
            self.layers = []
            self.layers.append(nn.Linear(4, hidden_dim, bias=bias))
            self.layers.append(nn.GELU(approximate="tanh"))
            for _ in range(num_layers - 2): 
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                self.layers.append(nn.GELU(approximate="tanh"))
            self.layers.append(nn.Linear(hidden_dim, 1))
            self.mlp = nn.Sequential(*self.layers)

    def forward(self, x, temb=None, foveation_mask=None):
        bsz, ch, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0
        
        # 1. Base: Standard interpolation everywhere (this covers the HR/Fovea regions)
        output = F.interpolate(x, scale_factor=0.5, mode="nearest") 
        
        # 2. Logic Change: Target LR regions (background)
        is_lr = foveation_mask < 0.5 
        
        # Prepare patches: [bsz, ch, h/2, w/2, 4]
        x = x.view(bsz, ch, h // 2, 2, w // 2, 2) 
        x = x.permute(0, 1, 2, 4, 3, 5).reshape(bsz, ch, h // 2, w // 2, 4) 
        
        # Select only LR blocks for MLP processing
        x_lr = x[:, :, is_lr, :] 
        x_lr = x_lr.reshape(-1, 4)
        
        # Apply MLP (Learned Downsampling) to LR regions
        x_lr_downsampled = self.mlp(x_lr) 
        x_lr_downsampled = x_lr_downsampled.reshape(bsz, ch, -1) 
        
        # Inject learned latents back into output
        output[:, :, is_lr] = x_lr_downsampled
        return output

class TokenUpsampler(nn.Module):
    def __init__(self, num_layers=2, hidden_dim=64, bias=False, debug=False):
        super().__init__()
        self.debug = debug

        if self.debug:
            # Debug: Replicate pixel 4 times (Nearest Neighbor equivalent for Upsample)
            self.mlp = nn.Linear(1, 4, bias=False)
            with torch.no_grad():
                #self.mlp.weight.fill_(1.0)
                self.mlp.weight.fill_(0)
                self.mlp.weight[0, 0] = 1.0
        else:
            self.layers = []
            self.layers.append(nn.Linear(1, hidden_dim, bias=bias))
            self.layers.append(nn.GELU(approximate="tanh"))
            for _ in range(num_layers - 2): 
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                self.layers.append(nn.GELU(approximate="tanh"))
            self.layers.append(nn.Linear(hidden_dim, 4))
            self.mlp = nn.Sequential(*self.layers)

    def forward(self, x, temb=None, foveation_mask=None):
        bsz, ch, h, w = x.shape
        
        # 1. Base: Standard interpolation everywhere (covers HR/Fovea)
        output = F.interpolate(x, scale_factor=2, mode="nearest") 
        output = output.reshape(bsz, ch, h, 2, w, 2).permute(0, 1, 2, 4, 3, 5).reshape(bsz, ch, h, w, 4)
        
        # 2. Logic Change: Target LR regions (background)
        is_lr = foveation_mask < 0.5 
        
        if is_lr.any():
            x_lr = x[:, :, is_lr].reshape(-1, 1) 
            
            # Apply MLP (Learned Upsampling) to LR regions
            x_lr_upsampled = self.mlp(x_lr) 
            x_lr_upsampled = x_lr_upsampled.reshape(bsz, ch, -1, 4) 
            
            # Inject learned latents back into output
            output[:, :, is_lr, :] = x_lr_upsampled
            
        output = output.reshape(bsz, ch, h, w, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(bsz, ch, 2 * h, 2 * w)
        return output

if __name__ == "__main__":
    # Setup dummy data for testing
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    foveation_mask = torch.zeros(64, 64).to(device)
    
    # Define Fovea (High Res) - Center Patch
    # Note: Logic is now inverted! 
    # MLP will process everything OUTSIDE this box (mask < 0.5)
    foveation_mask[32-10:32+10, 32-10:32+10] = 1 
    
    print(f"Running on {device}. Logic inverted: MLP processes background (mask=0).")

    # 1. Test downsampler
    try:
        x = read_image("results/flowers.png").to(device)
    except:
        x = torch.randn(1, 3, 128, 128).to(device) # Fallback if image missing
        
    print("Testing Downsampler (Debug Mode)...")
    downsampler = TokenDownsampler(num_layers=2, debug=True).to(device)
    x_downsampled = downsampler(x, temb=None, foveation_mask=foveation_mask)
    # x_downsampled regions outside fovea should look 'nearest neighbor' downsampled
    
    # 2. Test upsampler
    try:
        x_small = read_image("results/flowers.png", size=(64, 64)).to(device)
    except:
        x_small = torch.randn(1, 3, 64, 64).to(device)

    print("Testing Upsampler (Debug Mode)...")
    upsampler = TokenUpsampler(num_layers=2, debug=True).to(device)
    x_upsampled = upsampler(x_small, temb=None, foveation_mask=foveation_mask)
    # x_upsampled regions outside fovea should look 'nearest neighbor' upsampled
    
    print("Done.")