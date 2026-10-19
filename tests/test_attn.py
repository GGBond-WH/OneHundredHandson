import torch

from week03_attention.attn_loop import attention_loop, casual_mean_loop


def test_casual_mean():
    x = torch.randn(2, 4, 5)
    B, T, C = x.shape
    out = casual_mean_loop(x)
    assert x.shape == out.shape

    arange = torch.arange(1, T + 1, device=x.device).view(1, T, 1)
    expected = torch.cumsum(x, dim=1) / arange
    torch.testing.assert_close(out, expected)


def test_casual():
    B, T, C = 2, 6, 4
    x = torch.randn(B, T, C)
    wq = torch.randn(C, C)
    wk = torch.randn(C, C)
    wv = torch.randn(C, C)

    out = attention_loop(x, wq, wk, wv)
    t = 2
    x[:, t + 1 :] = torch.randn_like(x[:, t + 1 :])
    out2 = attention_loop(x, wq, wk, wv)
    torch.testing.assert_close(out[:, : t + 1], out2[:, : t + 1])
