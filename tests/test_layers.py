import torch

from week04_gpt.layers import MLP, LayerNorm


def test_LN():
    torch.manual_seed(0)
    B, T, C = 2, 4, 8
    x = torch.randn(B, T, C)
    LayerNorm_ = LayerNorm(C)
    y = LayerNorm_(x)  # y.shape: [B, T, C]
    u_y = y.mean(dim=-1, keepdim=True)
    v_y = y.var(dim=-1, correction=0, keepdim=True)
    u_tar = torch.zeros(B, T, 1)
    v_tar = torch.ones(B, T, 1)
    torch.testing.assert_close(u_y, u_tar)
    torch.testing.assert_close(v_y, v_tar, atol=1e-4, rtol=0)


def test_MLP():
    torch.manual_seed(0)
    B, T, C = 2, 4, 8
    x = torch.randn(B, T, C)
    MLP_ = MLP(C)
    output = MLP_(x)
    assert output.shape == x.shape
