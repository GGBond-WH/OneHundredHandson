from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class BigramLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        nn.init.normal_(self.embedding.weight, mean=0.00, std=0.02)
        self.flatten = nn.Flatten(0, 1)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ):  # idx shape:[B, T], targets shape:[B, T]
        logits = self.embedding(idx)  # model shape:[B, T, V]
        if targets is None:
            return (logits, None)
        logits_flatten = self.flatten(logits)
        targets_flatten = self.flatten(targets)
        loss = F.cross_entropy(logits_flatten, targets_flatten)
        return (logits, loss)


def save_checkpoint(
    model: BigramLM, vocab_size: int, chars: list[str], path: Path
) -> None:
    ckpt = {"model": model.state_dict(), "vocab_size": vocab_size, "chars": chars}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path: Path | str) -> BigramLM:
    ckpt = torch.load(path, weights_only=True)
    vocab_size = ckpt["vocab_size"]
    new_model = BigramLM(vocab_size)
    new_model.load_state_dict(ckpt["model"])
    return new_model
