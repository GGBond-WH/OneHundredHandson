import pytest

from week01_text_pipeline.text_pipeline.dataset import TextDataset, batch_iterator


@pytest.fixture
def dataset():
    ids = [0, 1, 2, 3, 4]
    block_size = 2
    return TextDataset(ids, block_size)


def test_dislocation(dataset):
    assert dataset[0] == ([0, 1], [1, 2])


def test_dataset_len(dataset):
    assert len(dataset) == 3


def test_iterator_order(dataset):
    a = list(batch_iterator(dataset, 3, seed=1))
    b = list(batch_iterator(dataset, 3))
    assert a != b
    a = list(batch_iterator(dataset, 3))
    b = list(batch_iterator(dataset, 3))
    print(a)
    assert a == b


def test_ids_len_short():
    with pytest.raises(ValueError):
        TextDataset([0], 2)


def test_batches_no_shuffle():
    ids = [0, 1, 2, 3, 4, 5, 6]
    block = 2
    batch = 2
    dataset = TextDataset(ids, block)
    assert list(batch_iterator(dataset, batch, shuffle=False)) == [
        [([0, 1], [1, 2]), ([1, 2], [2, 3])],
        [([2, 3], [3, 4]), ([3, 4], [4, 5])],
    ]


def test_index_error():
    ids = [0, 1, 2, 3, 4, 5, 6]
    block = 2
    dataset = TextDataset(ids, block)
    with pytest.raises(IndexError):
        dataset[5]
        dataset[-1]
