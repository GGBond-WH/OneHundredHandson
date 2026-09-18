import random


class TextDataset:
    def __init__(self, ids: list[int], block_size: int) -> None:
        """
        ids列表太短，不足以构建block_size体量，直接报ValueError.
        """
        self.ids = ids
        self.block_size = block_size
        if len(ids) < block_size + 1:
            raise ValueError("ids至少要有block_size+1个token")

    def __len__(self) -> int:
        return len(self.ids) - self.block_size

    def __getitem__(self, i: int) -> tuple[list[int], list[int]]:
        if i >= 0 and i + self.block_size + 1 <= len(self.ids):
            return (
                self.ids[i : i + self.block_size],
                self.ids[i + 1 : i + 1 + self.block_size],
            )
        else:
            raise IndexError("out of index!")


def batch_iterator(
    dataset: TextDataset, batch_size: int, shuffle: bool = True, seed: int = 0
):

    index = [i for i in range(len(dataset))]
    random.Random(seed)
    if shuffle:
        random.shuffle(index)
    for i in range(0, len(index), batch_size):
        batch = []
        for j in range(i, i + batch_size):
            batch.append(dataset[index[j]])
        yield batch
