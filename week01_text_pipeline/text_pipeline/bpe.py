def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    if len(ids) <= 1:
        raise ValueError("输入的id列表长度最少为2才能展开bpe算法")
    left = 0
    right = 1
    merge_dict = {}
    while right < len(ids):
        if (ids[left], ids[right]) not in merge_dict:
            merge_dict[(ids[left], ids[right])] = 1
        else:
            merge_dict[(ids[left], ids[right])] += 1
        left += 1
        right += 1
    sort_dict = dict(
        sorted(merge_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)
    )
    return sort_dict


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merge_list = []
    i = 0
    while i < len(ids):
        if i == len(ids) - 1:
            merge_list.append(ids[i])
            i += 1
        elif (ids[i], ids[i + 1]) == pair:
            merge_list.append(new_id)
            i += 2
        else:
            merge_list.append(ids[i])
            i += 1
    return merge_list


class BasicTokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}

    def train(self, text: str, vocab_size: int) -> None:
        for i in range(256):
            self.vocab[i] = bytes([i])
        ids = list(text.encode("utf-8"))
        while len(self.vocab) < vocab_size:
            max_pair = get_stats(ids=ids)
            pair = next(iter(max_pair))
            new_id = len(self.vocab)
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> list[int]:
        """
        输入的字符串utf-8编码并转化为list[int]列表
        外层循环标志：是否还有可合并项，没有的话直接结束
        内层循环：每次都遍历整个ids列表，查看vocab中是否还存在
        """
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        ids = list(text.encode("utf-8"))
        sign = True
        while sign:
            i = 0
            new_sign = False
            merge_list = []
            while i < len(ids):
                if i == len(ids) - 1:
                    merge_list.append(ids[i])
                    i += 1
                elif self.vocab[ids[i]] + self.vocab[ids[i + 1]] in reverse_vocab:
                    new_sign = True
                    merge_list.append(
                        reverse_vocab[self.vocab[ids[i]] + self.vocab[ids[i + 1]]]
                    )
                    i += 2
                else:
                    merge_list.append(ids[i])
                    i += 1
            if not new_sign:
                sign = False
            ids = merge_list
        return merge_list

    def decode(self, ids: list[int]) -> str:
        bytes_str = b""
        for id in ids:
            bytes_str += self.vocab[id]
        decode_str = bytes_str.decode("utf-8", errors="replace")
        return decode_str
