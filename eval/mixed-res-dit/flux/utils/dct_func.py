import math
import torch


_dct_basis_cache = {}


def get_dct_basis(n: int, device, dtype):
    key = (n, str(device), str(dtype))
    if key in _dct_basis_cache:
        return _dct_basis_cache[key]

    x = torch.arange(n, device=device, dtype=dtype)
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)

    basis = torch.cos(math.pi / n * (x + 0.5) * k)  # [n, n]
    basis[0] *= math.sqrt(1.0 / n)
    basis[1:] *= math.sqrt(2.0 / n)

    _dct_basis_cache[key] = basis
    return basis


def dct2_patches(x: torch.Tensor, dct_mat: torch.Tensor) -> torch.Tensor:
    """
    x: [..., P, P]
    dct_mat: [P, P]
    return: [..., P, P]
    """
    # DCT-II: Y = C X C^T
    return dct_mat @ x @ dct_mat.transpose(-1, -2)


@torch.no_grad()
def patch_dct_energy_mask(
    img: torch.Tensor,
    patch_size: int = 16,
    threshold: float = 0.15,
    low_freq_keep: int = 4,
    use_luma: bool = True,
):
    """
    Args:
        img: [B, 3, H, W], value range [-1, 1]
        patch_size: patch size, default 16
        threshold: fixed threshold for high-frequency energy ratio
        low_freq_keep: top-left low-frequency square size to exclude from "high freq"
        use_luma: convert RGB to grayscale luma before DCT

    Returns:
        score_map: [B, 1, H//P, W//P], float
        mask:      [B, 1, H//P, W//P], bool
    """
    assert img.ndim == 4 and img.shape[1] == 3, f"Expected [B,3,H,W], got {img.shape}"
    B, C, H, W = img.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, "H and W must be divisible by patch_size"

    device = img.device
    dtype = img.dtype

    # 1) RGB -> grayscale
    if use_luma:
        # works for [-1,1] too; linear combination is fine
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        x = 0.299 * r + 0.587 * g + 0.114 * b   # [B,1,H,W]
    else:
        # simple average if you don't want luma
        x = img.mean(dim=1, keepdim=True)

    # 2) split into non-overlapping patches
    # [B,1,H,W] -> [B,1,Hp,Wp,P,P]
    Hp, Wp = H // P, W // P
    patches = x.unfold(2, P, P).unfold(3, P, P)   # [B,1,Hp,Wp,P,P]
    patches = patches.contiguous().view(B, Hp, Wp, P, P)

    # 3) 2D DCT per patch
    Cmat = get_dct_basis(P, device=device, dtype=dtype)
    dct = dct2_patches(patches, Cmat)  # [B,Hp,Wp,P,P]

    # 4) energy
    e = dct.square()

    # exclude DC from total energy
    total = e.sum(dim=(-1, -2)) - e[..., 0, 0]

    # define high-frequency area by excluding top-left low-frequency block
    hf_mask = torch.ones((P, P), device=device, dtype=torch.bool)
    hf_mask[:low_freq_keep, :low_freq_keep] = False

    # also exclude DC explicitly
    hf_mask[0, 0] = False

    high = e[..., hf_mask].sum(dim=-1)

    score = high / (total + 1e-8)   # [B,Hp,Wp]
    score_map = score.unsqueeze(1)  # [B,1,Hp,Wp]

    mask = score_map > threshold

    return score_map, mask
