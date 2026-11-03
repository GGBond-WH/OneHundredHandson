import pytest
import torch
from torch.nn import functional as F

from week03_attention.CausalSelfAttention import CausalSelfAttention


@pytest.mark.parametrize(
    "B, T, C, H",
    [
        (1, 4, 32, 4),
        (2, 8, 64, 8),
        (4, 16, 32, 2),
    ],
)
def test_shape(B, T, C, H):
    x = torch.randn(B, T, C)
    model = CausalSelfAttention(C, H, 32)
    out = model(x)
    assert out.shape == (B, T, C)


def test_sdpa_reference():
    B, T, C, H = 2, 8, 64, 8
    x = torch.randn(B, T, C)
    model = CausalSelfAttention(C, H, 16)
    out = model(x)

    q, k, v = model.Linear3C(x).chunk(3, dim=-1)
    q = q.view(B, T, H, C // H).transpose(1, 2)
    k = k.view(B, T, H, C // H).transpose(1, 2)
    v = v.view(B, T, H, C // H).transpose(1, 2)
    ref_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref_out = model.Linearend(ref_attn.transpose(1, 2).contiguous().view(B, T, C))
    torch.testing.assert_close(out, ref_out)


def test_multihead_causality():
    B, T, C, H = 2, 8, 64, 8
    x = torch.randn(B, T, C)
    model = CausalSelfAttention(C, H, 16)
    out = model(x)

    t = 3
    x2 = x.clone()
    x2[:, t + 1 :] = torch.rand_like(x2[:, t + 1 :])
    out2 = model(x2)

    torch.testing.assert_close(out[:, : t + 1], out2[:, : t + 1])
