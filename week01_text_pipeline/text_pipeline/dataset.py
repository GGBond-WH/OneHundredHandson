class TextDataset:
    def __init__(self, ids: list[int], block_size: int) -> None:
        self.ids = ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.ids) - self.block_size

    def __getitem__(self, i: int) -> tuple[list[int], list[int]]:
        if i + self.block_size + 1 <= len(self.ids):
            return (
                self.ids[i : i + self.block_size],
                self.ids[i + 1 : i + 1 + self.block_size],
            )
        else:
            raise ValueError("out of index!\n")


def batch_iterator(
    dataset: TextDataset, batch_size: int, shuffle: bool = True, seed: int = 0
):
    for i in range(batch_size):
        yield dataset[i]
