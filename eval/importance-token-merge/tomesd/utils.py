import torch 

def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

def chunked_score_max(a, b, chunk_size=64):
    """
    Args
        a: B,M,C
        b: B,N,C
    Return
        max_scores, max_indices
    """
    B, M, C = a.shape
    B, N, C = b.shape

    max_scores = []
    max_indices = []

    for i in range(0, M, chunk_size):
        # Take a chunk of `a` of size (B, chunk_size, C)
        a_chunk = a[:, i:i+chunk_size, :]  # Shape: (B, chunk_size, C)
        
        # Compute the scores for this chunk only (B, chunk_size, N)
        scores_chunk = a_chunk @ b.transpose(-1, -2)

        # Find the max scores and indices for this chunk
        chunk_max, chunk_idx = scores_chunk.max(dim=-1)  # Shape: (B, chunk_size)

        max_scores.append(chunk_max)
        max_indices.append(chunk_idx)

    # Concatenate the results back along the `M` dimension
    max_scores = torch.cat(max_scores, dim=1)
    max_indices = torch.cat(max_indices, dim=1)

    return max_scores, max_indices


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def init_generator(device: torch.device, fallback: torch.Generator=None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback