"""faulty_kv — a tiny decoder-only LM with RoPE and pluggable KV caches.

Contract (what this module claims to do):

  1. `TinyLM.forward(idx, cache=None, window=None)` returns logits [B, T, V].
     - Without a cache, `idx` is the full sequence. With a cache, `idx` holds only the NEW tokens;
       everything seen since the last `reset()` lives in the cache.
     - Positions are absolute: the i-th new token has position `cache.seq_len + i` (RoPE uses it).
     - `window=W` (no cache) means sliding-window attention over the full sequence: a query at
       position t attends to keys at positions max(0, t-W+1) .. t — exactly min(t+1, W) keys, itself included.
  2. Prefill + token-by-token decode through a `DynamicCache` gives the same logits as one full
     forward (float tolerance), and identical greedy tokens.
  3. Prefilling in several chunks gives the same result as prefilling in one chunk.
  4. After `reset()` a cache behaves exactly like a freshly constructed one.
  5. `SlidingWindowCache(window=W)` reproduces the `window=W` semantics of (1) and never stores
     more than W positions per layer. `seq_len` always counts ALL tokens processed since reset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class Config:
    vocab_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 32
    max_seq_len: int = 128
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


# ---------------------------------------------------------------- caches
class DynamicCache:
    """Grows by concatenation. Stored K/V per layer: [B, H, T, D]."""

    window: int | None = None

    def __init__(self, n_layer: int) -> None:
        self.n_layer = n_layer
        self._k: list[Tensor | None] = [None] * n_layer
        self._v: list[Tensor | None] = [None] * n_layer
        self._len = 0  # tokens processed since reset (absolute count)

    @property
    def seq_len(self) -> int:
        return self._len

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Append this step's k/v ([B,H,T,D]) for `layer`; return the K/V that attention should use."""
        t_new = k.shape[2]
        if self._k[layer] is not None:
            k = torch.cat([k, self._k[layer]], dim=2)
            v = torch.cat([v, self._v[layer]], dim=2)
        self._k[layer], self._v[layer] = self._trim(k), self._trim(v)
        if layer == self.n_layer - 1:  # count once per forward, after the last layer
            self._len += t_new
        return k, v

    def get(self, layer: int) -> tuple[Tensor | None, Tensor | None]:
        return self._k[layer], self._v[layer]

    def reset(self) -> None:
        self._k = [None] * self.n_layer
        self._v = [None] * self.n_layer

    def _trim(self, x: Tensor) -> Tensor:
        return x


class SlidingWindowCache(DynamicCache):
    """Stores only the most recent `window` positions of K/V per layer."""

    def __init__(self, n_layer: int, window: int) -> None:
        super().__init__(n_layer)
        self.window = window

    def _trim(self, x: Tensor) -> Tensor:
        return x[:, :, -self.window :] if x.shape[2] > self.window else x


# ---------------------------------------------------------------- model parts
def rope_tables(cfg: Config) -> tuple[Tensor, Tensor]:
    d = cfg.head_dim
    inv = 1.0 / (cfg.rope_base ** (torch.arange(0, d, 2).float() / d))  # [D/2]
    t = torch.arange(cfg.max_seq_len).float()
    freqs = torch.outer(t, inv)  # [max_seq_len, D/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, positions: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: [B, H, T, D]; positions: [T] absolute positions of the T tokens."""
    T = x.shape[2]
    c = cos[:T][None, None]  # [1, 1, T, D/2]
    s = sin[:T][None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def build_mask(q_pos: Tensor, k_pos: Tensor, window: int | None) -> Tensor:
    """bool [T, K]; True = query may attend key. Causal, plus a sliding window if given."""
    allowed = k_pos[None, :] <= q_pos[:, None]
    if window is not None:
        allowed &= k_pos[None, :] >= q_pos[:, None] - window
    return allowed


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: Config, layer: int) -> None:
        super().__init__()
        self.cfg, self.layer = cfg, layer
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: Tensor, positions: Tensor, cos: Tensor, sin: Tensor, cache, window: int | None) -> Tensor:
        B, T, C = x.shape
        H, D = self.cfg.n_head, self.cfg.head_dim
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        q = apply_rope(q, positions, cos, sin)
        k = apply_rope(k, positions, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer, k, v)  # [B, H, K, D], K = stored + T
        kv_len = k.shape[2]
        end = int(positions[-1]) + 1  # absolute position just past the newest token
        k_pos = torch.arange(end - kv_len, end, device=x.device)
        mask = build_mask(positions, k_pos, window)  # [T, K]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, K]
        scores = scores.masked_fill(~mask, float("-inf"))
        out = scores.softmax(-1) @ v  # [B, H, T, D]
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: Config, layer: int) -> None:
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.n_embd), RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg, layer)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(), nn.Linear(4 * cfg.n_embd, cfg.n_embd))

    def forward(self, x, positions, cos, sin, cache, window):
        x = x + self.attn(self.n1(x), positions, cos, sin, cache, window)
        return x + self.mlp(self.n2(x))


class TinyLM(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx: Tensor, cache: DynamicCache | None = None, window: int | None = None) -> Tensor:
        """idx: [B, T] — the full sequence (no cache) or only the new tokens (with cache)."""
        B, T = idx.shape
        past_len = 0 if cache is None else cache.seq_len
        if cache is not None:
            window = cache.window
        positions = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x, positions, self.cos, self.sin, cache, window)
        return self.head(self.norm(x))


@torch.no_grad()
def generate(model: TinyLM, idx: Tensor, max_new_tokens: int, cache: DynamicCache | None = None) -> Tensor:
    """Greedy decoding. With a cache: prefill once, then feed one token at a time."""
    model.eval()
    if cache is None:
        for _ in range(max_new_tokens):
            logits = model(idx[:, -model.cfg.max_seq_len :])
            idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return idx
    cache.reset()
    logits = model(idx, cache)
    for _ in range(max_new_tokens):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        logits = model(nxt, cache)
    return idx
