from pathlib import Path

import pytest
from torch import optim

from week01_text_pipeline.text_pipeline.dataset import TextDataset
from week01_text_pipeline.text_pipeline.tokenizer import CharTokenizer
from week02_bigram.mini_lm.train import overfit_one_batch, train_loop
from week04_gpt.model import GPT


@pytest.fixture
def setup_train_data():
    n_layer = 4
    n_head = 4
    n_embd = 128
    block_size = 64
    batch = 32
    lr = 3e-4
    device = "mps"

    text = Path("data/input.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text=text)
    vocab_size = tokenizer.vocab_size
    ids = tokenizer.encode(text)
    split = int(len(ids) * 0.9)
    train_ids = ids[:split]
    train_dataset = TextDataset(train_ids, block_size)
    model = GPT(vocab_size, n_embd, n_head, block_size, n_layer)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    return model, optimizer, train_dataset, batch, device


def test_tiny_gpt(setup_train_data):
    model, optimizer, train_dataset, batch, device = setup_train_data
    origin_loss, val_loss = overfit_one_batch(
        model, optimizer, train_dataset, batch, device
    )
    assert val_loss < origin_loss * 0.6


def test_smoke(setup_train_data):
    model, optimizer, train_dataset, batch, device = setup_train_data
    device = "cpu"
    train_loop(
        model,
        optimizer=optimizer,
        iter_num=5,
        train_dataset=train_dataset,
        device=device,
        batch_size=batch,
    )
