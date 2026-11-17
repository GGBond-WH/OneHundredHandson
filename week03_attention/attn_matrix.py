import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    tril: torch.Tensor

    def __init__(self, c_in: int, d_out: int, max_seq_len: int = 512) -> None:
        super().__init__()
        self.wq = nn.Linear(c_in, d_out, bias=False)
        self.wk = nn.Linear(c_in, d_out, bias=False)
        self.wv = nn.Linear(c_in, d_out, bias=False)
        mask = torch.ones(max_seq_len, max_seq_len)
        mask = torch.tril(mask)
        self.register_buffer("tril", mask)

    def forward(self, x: torch.Tensor):
        # x.shape: [B, T, C]
        # wq.shape: [C, D]
        q = self.wq(x)  # q.shape: [B, T, D]
        k = self.wk(x)  # k.shape: [B, T, D]
        v = self.wv(x)  # v.shape: [B, T, D]
        out = self.com_loop(q, k, v)
        return out

    def com_loop(self, q, k, v):
        _, T, D = q.shape
        wei = q @ k.transpose(1, 2) * D**-0.5  # wei.shape: [B, T, T]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = torch.softmax(wei, dim=-1)
        out = wei @ v
        return out
