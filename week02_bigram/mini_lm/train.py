import torch
from torch import nn, optim

from week01_text_pipeline.text_pipeline.dataset import TextDataset


def get_batch(dataset: TextDataset) -> tuple[torch.Tensor, torch.Tensor]:
    batch_id = torch.randint(0, len(dataset), (32,))
    x_batch = torch.stack([dataset[int(i)][0] for i in batch_id])
    y_batch = torch.stack([dataset[int(i)][1] for i in batch_id])
    return x_batch, y_batch


def train_loop(
    model: nn.Module, optimizer: optim.AdamW, iter_num: int, train_dataset: TextDataset
) -> None:
    model.train()
    for i in range(iter_num):
        x_batch, y_batch = get_batch(train_dataset)
        _, loss = model(x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            loss = loss.item()
            print(f"train loss: {loss:>7f}\n")


def eval_loop(model: nn.Module, iter_num: int, val_dataset: TextDataset) -> None:
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for i in range(iter_num):
            x_batch, y_batch = get_batch(val_dataset)
            _, loss = model(x_batch, y_batch)
            eval_loss += loss

    eval_loss /= iter_num
    print(f"average loss: {eval_loss:>7f}\n")


def overfit_one_batch(
    model: nn.Module, optimizer: optim.AdamW, train_dataset: TextDataset
) -> tuple[float, float]:
    model.train()
    origin_loss = 0.0
    loss_val = 0.0
    x_batch, y_batch = get_batch(train_dataset)
    for i in range(200):
        _, loss = model(x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())
        if i == 0:
            origin_loss = loss_val
        if i % 100 == 0:
            print(f"step: {i:3d} | train loss: {loss:>7f}\n")
    return origin_loss, loss_val
