import torch
from torch import nn


@torch.no_grad
def generate(
    model: nn.Module, idx: torch.Tensor, max_new_tokens: int, greedy=False
) -> torch.Tensor:  # idx shape: (B, T)
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(idx)  # logits shape: (B, T, V)
        logits = logits[:, -1, :]  # logits shape: (B, V)
        softmax = nn.Softmax(dim=1)  # softmax输出与输入形状一致
        probability = softmax(logits)  # V个token，每个token概率
        if greedy:
            out = torch.argmax(probability, dim=1, keepdim=True)
        else:
            out = torch.multinomial(probability, 1)
        idx = torch.cat((idx, out), dim=1)  # idx shape: (B, V + max_new_tokens)
    return idx
