import math
from pathlib import Path

import pytest
import torch

from week01_text_pipeline.text_pipeline.dataset import TextDataset, batch_iterator
from week01_text_pipeline.text_pipeline.io import load_text
from week01_text_pipeline.text_pipeline.tokenizer import CharTokenizer
from week02_bigram.mini_lm.model import BigramLM


@pytest.fixture
def setup_data():
    # 用 __file__ 定位项目根目录下的 data/input.txt，避免工作目录不同导致找不到文件
    data_path = Path(__file__).resolve().parent.parent / "data" / "input.txt"
    init_str = load_text(data_path)
    tokenizer = CharTokenizer(init_str)
    vocab_size = tokenizer.vocab_size
    bigramlm = BigramLM(vocab_size=vocab_size)

    test_str = "LLM is the best product of the century."
    ids = tokenizer.encode(test_str)
    block_size = 3
    dataset = TextDataset(ids, block_size)
    batch_size = 3

    # 只取第 1 个 batch，保证第 0 维长度正好是 batch_size (3)
    first_batch = next(batch_iterator(dataset, batch_size))
    # 如果 batch_iterator 返回的是 [(x1, y1), (x2, y2), (x3, y3)]：
    idx = torch.stack([item[0] for item in first_batch])
    targets = torch.stack([item[1] for item in first_batch])
    # （注：如果你已经在 batch_iterator 内部做好了 torch.stack 返回 (x_batch, y_batch)，
    #   那上面三行直接写成 idx, targets = next(batch_iterator(dataset, batch_size)) 即可）

    return bigramlm, idx, targets, batch_size, block_size, vocab_size


def test_logits_shape(setup_data):
    bigramlm, idx, targets, batch_size, block_size, vocab_size = setup_data
    logits, _ = bigramlm(idx, targets)
    assert logits.shape == (batch_size, block_size, vocab_size)


def test_targets_none(setup_data):
    bigramlm, idx, _, _, _, _ = setup_data
    _, loss = bigramlm(idx, None)
    assert loss is None


def test_loss_value(setup_data):
    bigramlm, idx, targets, _, _, vocab_size = setup_data
    predict_loss = math.log(vocab_size)
    _, loss = bigramlm(idx, targets)
    assert predict_loss - 0.1 <= loss <= predict_loss + 0.1
