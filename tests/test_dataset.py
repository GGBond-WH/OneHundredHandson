import pytest

from week01_text_pipeline.text_pipeline.dataset import TextDataset, batch_iterator


@pytest.fixture
def get_dataset():
    ids = [0, 1, 2, 3, 4]
    block_size = 2
    return TextDataset(ids, block_size)


def test_dislocation(get_dataset):
    assert get_dataset[0] == ([0, 1], [1, 2])


def test_dataset_len(get_dataset):
    assert len(get_dataset) == 3


def test_iterator_order(get_dataset):
    a = list(batch_iterator(get_dataset, 3, seed=1))
    b = list(batch_iterator(get_dataset, 3))
    assert a != b
    a = list(batch_iterator(get_dataset, 3))
    b = list(batch_iterator(get_dataset, 3))
    print(a)
    assert a == b


def test_ids_len_short():
    with pytest.raises(ValueError):
        TextDataset([0], 2)
