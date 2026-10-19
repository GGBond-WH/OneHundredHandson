from pathlib import Path

import pytest
import torch
from torch import optim

from week01_text_pipeline.text_pipeline.dataset import TextDataset
from week01_text_pipeline.text_pipeline.io import load_text
from week01_text_pipeline.text_pipeline.tokenizer import CharTokenizer
from week02_bigram.mini_lm.generate import generate
from week02_bigram.mini_lm.model import BigramLM, load_checkpoint, save_checkpoint
from week02_bigram.mini_lm.train import train_loop


@pytest.fixture
def setup_train_data():
    data_path = Path(__file__).resolve().parent.parent / "data" / "input.txt"
    data_str = load_text(data_path)
    tokenizer = CharTokenizer(data_str)
    vocab_size = tokenizer.vocab_size
    data_ids = tokenizer.encode(data_str)
    train_data_ids = data_ids[: int((len(data_ids) * 9) / 10)]
    val_data_ids = data_ids[int((len(data_ids) * 9) / 10) :]
    block_size = 8
    train_dataset = TextDataset(train_data_ids, block_size)
    val_dataset = TextDataset(val_data_ids, block_size)
    iter_num = 3000
    model = BigramLM(vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    train_loop(model, optimizer, iter_num, train_dataset)
    chars = list(tokenizer.vocab.keys())
    return model, vocab_size, chars


def test_checkpoint_roundtrip(setup_train_data):
    model, vocab_size, chars = setup_train_data
    path = Path(__file__).resolve().parent.parent / "checkpoint" / "ckpt.pt"
    save_checkpoint(model, vocab_size, chars, path)
    new_model = load_checkpoint(path)
    idx = torch.tensor([[10, 25, 30]])
    logits1, _ = model(idx)
    logits2, _ = new_model(idx)
    assert torch.equal(logits1, logits2)


def test_generate_len(setup_train_data):
    model, _, _ = setup_train_data
    idx = torch.tensor([[10, 25, 30]])
    max_new_tokens = 20
    out = generate(model, idx, max_new_tokens, True)
    assert out.shape[1] == idx.shape[1] + max_new_tokens
