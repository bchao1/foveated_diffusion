import torch 
from typing import Optional, Tuple, Callable

from .utils import do_nothing, mps_gather_workaround, chunked_score_max

def matching_by_heat_map_dit(metric: torch.Tensor,
        w: int, h: int, r: int,
        generator: torch.Generator = None,
        error_mask = None,
        kk: float = 0.25,
        pp: float = 1.0,
        dst_mode: str = "random",
        add_rand_small: bool = True,
        original_tome: bool = False) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - r: number of tokens to remove (by merging)
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        heat_map = error_mask['heat_map']

        B0 = heat_map.shape[0] 
        assert len(heat_map.shape) == 3 # B//2 h w 
        assert heat_map.shape[1] == h and heat_map.shape[2] == w
        assert N == w*h
        assert B0 == B//2
        assert r <= N

        num_dst = int( N * kk )
        M = int(  (N-r)  *  pp )
        M = min(M, N)
        assert M >= num_dst
        # print(f"num_dst={num_dst}  M={M}")

        # 1. select set0 - important tokens
        heat_list = heat_map.reshape(B0, -1, 1) # B//2, h*w, 1
        heat_idx = error_mask['heat_idx'] # heat_list.argsort(dim=1, descending=True) # B//2, h*w, 1
        assert heat_list.shape[1] == N
        set0_idx = heat_idx[:, :M] # B//2, M, 1
        heat_thr = torch.gather(heat_list, dim=1, index=set0_idx[:, -1:, :]) # B//2, 1, 1
        # print("set0_idx.shape", set0_idx.shape)

        # 2. select dst
        if dst_mode == "random":
            tmp_idx = torch.stack([
                torch.randperm(M)[:num_dst] for _ in range(B0)
            ]).unsqueeze(-1).to(set0_idx.device)
            dst_idx = torch.gather(set0_idx, dim=1, index=tmp_idx) # B//2, K, 1
        elif dst_mode == "top_k":
            dst_idx = set0_idx[:, :num_dst] # B//2, K, 1
        elif dst_mode == "bottom_k":
            dst_idx = set0_idx[:, -num_dst:] # B//2, K, 1
        elif dst_mode == "evenly_k":
            tmp_idx = set0_idx[:, ::(M//num_dst)] 
            dst_idx = tmp_idx[:, :num_dst] # B//2, K, 1
        else:
            raise ValueError(f"Unknown dst_mode: {dst_mode}")

        # assert uniqueness in dst_idx 

        # 3. get non_dst_idx
        tmp_mask = torch.zeros(B0, N, dtype=torch.bool, device=dst_idx.device) # B//2, N
        tmp_mask.scatter_(1, dst_idx.squeeze(-1), True)
        all_indices = torch.arange(N, device=dst_idx.device).unsqueeze(0) # 1, N
        all_indices = all_indices.expand(B0, -1) # B//2, N
        non_dst_idx = all_indices.masked_select(~tmp_mask).view(B0, N - num_dst, 1) # B//2, N-K, 1
        # _non_dst_idx = torch.stack([all_indices[b][~tmp_mask[b]] for b in range(B0)]).unsqueeze(-1)
        
        # 4. copy -> unconditional & conditional
        a_idx = non_dst_idx
        b_idx = dst_idx
        if B0 < B:
            assert B0 * 2 == B
            a_idx = torch.cat([a_idx]*2) # B, N-K, 1
            b_idx = torch.cat([b_idx]*2) # B, K, 1
            heat_thr = torch.cat([heat_thr]*2) # B, 1, 1
            heat_list = torch.cat([heat_list]*2) # B, N, 1
        assert a_idx.shape == (B, N - num_dst, 1) and b_idx.shape == (B, num_dst, 1)  

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(-1, -1, C)) # gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(-1, -1, C)) # gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # 5. Cosine similarity
        if original_tome:
            metric = metric / metric.norm(dim=-1, keepdim=True) # B, h*w, C
        a, b = split(metric) 
        
        # Can't reduce more than the # tokens in src
        assert a.shape[1] >= r

        # 6. Draw an edge from each a[i] 
        if original_tome:
            scores = a @ b.transpose(-1, -2)
        else:
            scores = - torch.cdist(a.float(), b.float(), p=2)
        node_max, node_idx = scores.max(dim=-1) # for each src token, find its most similar dst token

        if add_rand_small and not original_tome:
            # some small randomness may be a bit better
            small_randomness = torch.rand_like(node_max[:B//2]) * 1e-5
            small_randomness = torch.cat([small_randomness,]*2)
            node_max = node_max + small_randomness

        # 7. un-important tokens: set node_max -> 10000, -> always src
        a_heat = gather(heat_list, dim=1, index=a_idx) # == split(heat_list)[0] # B, N-K, 1
        node_heat_mask = (a_heat < heat_thr).squeeze(-1)
        node_max[node_heat_mask] += 100000 # B, N-K

        # 8. split a -> unm, src
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # B, N-K, 1
        
        unm_idx = edge_idx[..., r:, :]  # B, 6912-r, 1 UnMerged Tokens: low similarity with all dst token, later, no merge
        src_idx = edge_idx[..., :r, :]  # B, r, 1      Merged Tokens: high similarity with a dst token, later, merge it with dst
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx) # B, 6912, 1

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        if mode != "none":
            src = gather(src, dim=-2, index=src_idx.expand(n, r, c)) # B, 6912, C
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        # print("merge shapes", unm.shape, dst.shape, src.shape, dst_idx.shape)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge


def bipartite_soft_matching_random2d_dit(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None,
                                     add_rand_small: bool = True,
                                     original_tome: bool = False) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        # print(metric.shape)
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1) # 1, 96*96, 1

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        if original_tome:
            metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)

        # Can't reduce more than the # tokens in src
        assert a.shape[1] >= r
        r = min(a.shape[1], r)

        # Find the most similar greedily 
        if original_tome:
            scores = a @ b.transpose(-1, -2) # B, 6912, 2304
        else:
            scores = - torch.cdist(a.float(), b.float(), p=2)
        node_max, node_idx = scores.max(dim=-1) # for each src token, find its most similar dst token
        
        if add_rand_small and not original_tome:
            # some small randomness may be a bit better
            small_randomness = torch.rand_like(node_max[:B//2]) * 1e-5
            small_randomness = torch.cat([small_randomness,]*2)
            node_max = node_max + small_randomness

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # 2B, 3072, 1

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens: low similarity with all dst token, later, no merge
        src_idx = edge_idx[..., :r, :]  # Merged Tokens: high similarity with a dst token, later, merge it with dst
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx) # 2B, 6912, 1

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        if mode != "none":
            src = gather(src, dim=-2, index=src_idx.expand(n, r, c)) # 2, 6912, C
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        # print(unm.shape, dst.shape, src.shape, dst_idx.shape)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge
