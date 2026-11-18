import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x):
        eps = 1e-5
        _, _, x_embd = x.shape
        assert x_embd == self.n_embd
        u = x.mean(dim=-1, keepdim=True)
        v = x.var(dim=-1, correction=0, keepdim=True)
        x_bar = (x - u) / (torch.sqrt(v + eps))
        y = self.gamma * x_bar + self.beta
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.Linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.Linear2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.Linear1(x)
        x = self.gelu(x)
        x = self.Linear2(x)
        x = self.dropout(x)
        return x
