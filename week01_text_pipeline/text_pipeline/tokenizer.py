class CharTokenizer:
    """
    unk 的 id 是 len(self.vocab)-1；encode 遇到不在表里的字符返回 len(self.vocab)-1；decode 遇到不在表里的 id 返回"\ufffd"；vocab_size 等于 len(self.vocab)。
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab = {}
        self.re_vocab = {}
        for i, char in enumerate(chars):
            self.vocab[char] = i
            self.re_vocab[i] = char
        vocab_len = len(self.vocab)
        self.vocab["\ufffd"] = vocab_len
        self.re_vocab[vocab_len] = "\ufffd"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, s: str) -> list[int]:
        """
        编码字符串，转换为id列表。

        遇到vocab里面没有的字符，统一id为：len(self.vocab) - 1

        Args:
        参数1：需要进行编码的字符串s

        Returns：
        返回编码后的id列表

        """
        encode_list = []
        for char in s:
            if char not in self.vocab:
                encode_list.append(self.vocab["\ufffd"])
            else:
                encode_list.append(self.vocab[char])
        return encode_list

    def decode(self, ids: list[int]) -> str:
        """
        对ids列表展开解码，转换为原字符串。

        遇到id为len(self.vocab)-1以及不在re_vocab里面的id的id统一解码为：\ufffd

        Args:
        参数1：id列表

        Returns：
        解码后的字符串
        """
        decode_str = ""
        for id in ids:
            if id not in self.re_vocab:
                decode_str += "\ufffd"
            else:
                decode_str += self.re_vocab[id]
        return decode_str
