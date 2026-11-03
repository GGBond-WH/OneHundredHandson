import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """
    x = [B, T, C]首先一次性输出qkv = [B, T, 3C]
    然后将qkv切分出q, k, v: [B, T, C]
    沿着C将其切分为特定数目个头, q, k, v: [B, T, H, D]
    不同头的每个[B, T, H, D]需要计算出token之间的因果注意力权重, 批量矩阵乘法默认计算后两维
    [B, T, H, D]需要将T和H的维度交换: [B, H, T, D]计算注意力权重[B, H, T, T]
    按照这个注意力权重计算出注意力分数: [B, H, T, D]
    得到所有头的输出后, 拼接起来: [B, T, C]
    最后经过一个线性层[C, C]: [B, T, C]
    """

    tril: torch.Tensor

    def __init__(
        self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.D = n_embd // n_head
        self.block_size = block_size
        self.Linear3C = nn.Linear(n_embd, 3 * n_embd)
        self.Linearend = nn.Linear(n_embd, n_embd)
        mask = torch.ones(block_size, block_size)
        mask = torch.tril(mask)
        self.register_buffer("tril", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Linear3C(x)
        q, k, v = torch.chunk(x, 3, dim=-1)
        B, T, C = q.shape
        q = q.view(B, T, self.n_head, self.D).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.D).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.D).transpose(1, 2)  # [B, H, T, D]
        wei = q @ k.transpose(-1, -2) * self.D**-0.5  # [B, H, T, T]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = torch.softmax(wei, dim=-1)
        out = wei @ v  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, C)
        out = self.Linearend(out)  # [B, T, C]
        return out
