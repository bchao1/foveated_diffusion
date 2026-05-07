import torch, os
from einops import rearrange


FLASH_ATTN_3_AVAILABLE = False
FLASH_ATTN_2_AVAILABLE = False
SAGE_ATTN_AVAILABLE = False
XFORMERS_AVAILABLE = False
SPARGE_ATTN_2_AVAILABLE = False
SPARGE_ATTN_1_AVAILABLE = False

# try:
#     import flash_attn_interface
#     FLASH_ATTN_3_AVAILABLE = True
# except ModuleNotFoundError:
#     FLASH_ATTN_3_AVAILABLE = False

# try:
#     import flash_attn
#     FLASH_ATTN_2_AVAILABLE = True
# except ModuleNotFoundError:
#     FLASH_ATTN_2_AVAILABLE = False

# try:
#     from sageattention import sageattn
#     SAGE_ATTN_AVAILABLE = True
# except ModuleNotFoundError:
#     SAGE_ATTN_AVAILABLE = False

# try:
#     import xformers.ops as xops
#     XFORMERS_AVAILABLE = True
# except ModuleNotFoundError:
#     XFORMERS_AVAILABLE = False

# try:
#     from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
#     SPARGE_ATTN_2_AVAILABLE = True
# except (ModuleNotFoundError, ImportError):
#     SPARGE_ATTN_2_AVAILABLE = False

# try:
#     from spas_sage_attn import spas_sage_attn_meansim_topk_cuda
#     SPARGE_ATTN_1_AVAILABLE = True
# except (ModuleNotFoundError, ImportError):
#     SPARGE_ATTN_1_AVAILABLE = False

SPARGE_ATTN_AVAILABLE = SPARGE_ATTN_2_AVAILABLE or SPARGE_ATTN_1_AVAILABLE

def _is_ampere_gpu():
    """Check if the current GPU is Ampere (SM 8.x) which lacks fp8 support."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return major == 8
    return False

# On Ampere GPUs (A100, A6000, etc.), prefer the non-"2" variant (int8 QK + fp16 PV).
# On Ada/Hopper+ GPUs, prefer the "2" variant (fp8 PV).
USE_SPARGE_V1 = _is_ampere_gpu() and SPARGE_ATTN_1_AVAILABLE


def initialize_attention_priority():
    # print availability of attention implementations
    print("Attention implementation availability:")
    print("SPARGE Attention (v2/fp8):", SPARGE_ATTN_2_AVAILABLE)
    print("SPARGE Attention (v1/int8):", SPARGE_ATTN_1_AVAILABLE)
    print("SPARGE Attention using v1 (Ampere):", USE_SPARGE_V1)
    print("Flash Attention 3:", FLASH_ATTN_3_AVAILABLE)
    print("Flash Attention 2:", FLASH_ATTN_2_AVAILABLE)
    print("SAGE Attention:", SAGE_ATTN_AVAILABLE)
    print("xformers Attention:", XFORMERS_AVAILABLE)        
    env_impl = os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION')
    if env_impl is not None:
        env_impl = env_impl.lower()
        availability = {
            "sparge_attention": SPARGE_ATTN_AVAILABLE,
            "flash_attention_3": FLASH_ATTN_3_AVAILABLE,
            "flash_attention_2": FLASH_ATTN_2_AVAILABLE,
            "sage_attention": SAGE_ATTN_AVAILABLE,
            "xformers": XFORMERS_AVAILABLE,
            "torch": True,
        }
        if env_impl in availability and availability[env_impl]:
            return env_impl
        elif env_impl in availability:
            print(f"Warning: DIFFSYNTH_ATTENTION_IMPLEMENTATION='{env_impl}' is set but not available. Falling back to auto-detection.")
        else:
            print(f"Warning: DIFFSYNTH_ATTENTION_IMPLEMENTATION='{env_impl}' is not a recognized implementation. Falling back to auto-detection.")
    if FLASH_ATTN_3_AVAILABLE:
        return "flash_attention_3"
    elif FLASH_ATTN_2_AVAILABLE:
        return "flash_attention_2"
    elif SAGE_ATTN_AVAILABLE:
        return "sage_attention"
    elif XFORMERS_AVAILABLE:
        return "xformers"
    elif SPARGE_ATTN_AVAILABLE:
        return "sparge_attention"
    else:
        return "torch"


ATTENTION_IMPLEMENTATION = initialize_attention_priority()
print("Using attention implementation:", ATTENTION_IMPLEMENTATION)


def rearrange_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", required_in_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if q_pattern != required_in_pattern:
        q = rearrange(q, f"{q_pattern} -> {required_in_pattern}", **dims)
    if k_pattern != required_in_pattern:
        k = rearrange(k, f"{k_pattern} -> {required_in_pattern}", **dims)
    if v_pattern != required_in_pattern:
        v = rearrange(v, f"{q_pattern} -> {required_in_pattern}", **dims)
    return q, k, v


def rearrange_out(out: torch.Tensor, out_pattern="b n s d", required_out_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if out_pattern != required_out_pattern:
        out = rearrange(out, f"{required_out_pattern} -> {out_pattern}", **dims)
    return out


def torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None):
    required_in_pattern, required_out_pattern= "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=scale)
    if isinstance(out, tuple):
        out = out[0]
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn.flash_attn_func(q, k, v, softmax_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def sage_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = sageattn(q, k, v, sm_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def xformers_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = xops.memory_efficient_attention(q, k, v, scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def sparge_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None, topk=0.5):
    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    if USE_SPARGE_V1:
        out = spas_sage_attn_meansim_topk_cuda(q, k, v, topk=topk, is_causal=False)
    else:
        out = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=topk, is_causal=False)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, compatibility_mode=False, sparge_topk=0.5):
    if compatibility_mode or (attn_mask is not None):
        return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale)
    else:
        if ATTENTION_IMPLEMENTATION == "flash_attention_3":
            return flash_attention_3(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif ATTENTION_IMPLEMENTATION == "flash_attention_2":
            return flash_attention_2(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif ATTENTION_IMPLEMENTATION == "sage_attention":
            return sage_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif ATTENTION_IMPLEMENTATION == "sparge_attention":
            return sparge_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale, topk=sparge_topk)
        elif ATTENTION_IMPLEMENTATION == "xformers":
            return xformers_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        else:
            return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
