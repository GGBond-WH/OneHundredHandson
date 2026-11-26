import torch

from week04_gpt.model import GPT

B = 2
T = 4
C = 8
V = 32

n_head = 2
block_size = 8
N = 2


def test_model():
    tokens = torch.randint(0, V, (B, T + 1))
    idx = tokens[:, :-1]
    target = tokens[:, 1:]
    model = GPT(V, C, n_head, block_size, N)
    model.eval()
    logits, loss = model(idx, target)
    assert logits.shape == (B, T, V)
    assert loss.ndim == 0

    breakpt = 2
    idx_var = idx.clone()
    idx_var[:, breakpt:] = torch.randint(0, V, (B, T - breakpt))
    logits_var, _ = model(idx_var, target)
    torch.testing.assert_close(logits[:, :breakpt], logits_var[:, :breakpt])
