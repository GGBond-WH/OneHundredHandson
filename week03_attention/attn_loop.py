import math

import torch


def casual_mean_loop(x: torch.Tensor):
    B, T, _ = x.shape
    out = torch.zeros_like(x)
    for b in range(B):
        for t in range(T):
            out[b, t] = x[b, : t + 1].mean(dim=0)
    return out


def attention_loop(
    x: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor
) -> torch.Tensor:

    # wq shape: [C, D], wk shape: [C, D], wv shape: [C, D]
    q = x @ wq  # wq shape: [B, T, D]
    k = x @ wk  # wk shape: [B, T, D]
    v = x @ wv  # wv shape: [B, T, D]
    B, T, D = q.shape
    out = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    for b in range(B):
        for t in range(T):
            scores = (q[b, t] @ k[b, : t + 1].T) / math.sqrt(D)  # [D] @ [t+1, D].T
            weights = torch.softmax(scores, dim=-1)  # [t+1]
            out[b, t] = weights @ v[b, : t + 1]  # [t+1] @ [t+1, D] ---> [D]
    return out
