import torch
import torch.nn.functional as F
from torch import nn

from week03_attention.CausalSelfAttention import CausalSelfAttention
from week04_gpt.layers import MLP, LayerNorm


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)

    def forward(self, x):
        x_origin = x
        x = self.ln1(x)
        x = self.attn(x)
        h = x_origin + x
        h_origin = h
        h = self.ln2(h)
        h = self.mlp(h)
        h = h_origin + h
        return h


class GPT(nn.Module):
    def __init__(
        self, vocab_size: int, n_embd: int, n_head: int, block_size: int, N: int
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(N)]
        )
        self.ln_f = LayerNorm(n_embd)

    def forward(self, idx, target=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        T = idx.shape[1]
        positions = torch.arange(0, T, device=idx.device)
        token_emb = self.tok_emb(idx)  # token.shape: [B, T, C]
        position_emb = self.pos_emb(positions)  # position.shape: [T, C]
        fusion_emb = token_emb + position_emb  # fusion.shape: [B, T, C]
        for block in self.blocks:
            fusion_emb = block(fusion_emb)
        fusion_emb = self.ln_f(fusion_emb)
        logits = self.lm_head(fusion_emb)  # logits.shape: [B, T, V]
        V = logits.shape[-1]
        if target is not None:
            loss = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1))
        else:
            loss = None
        return (logits, loss)
