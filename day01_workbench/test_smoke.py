import torch


def test_torch_add():
    assert torch.tensor([1] + 1) == 2
