class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.text = text
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
    def vocab_size(
        self,
    ) -> int:
        return len(self.vocab)

    def encode(self, s: str) -> list[int]:
        """
        编码字符串，转换为id列表。

        遇到vocab里面没有的字符，统一id为：len(self.vocab)

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

        遇到id为len(self.vocab)的id统一解码为：\ufffd

        Args:
        参数1：id列表

        Returns：
        解码后的字符串
        """
        decode_str = ""
        for id in ids:
            decode_str += self.re_vocab[id]
        return decode_str
