from week01_text_pipeline.text_pipeline.tokenizer import CharTokenizer

s = "fhsetsfse. "
unknow_str = "🤩"
text = "abcdefghijkkkllllmnopqrstuvwxyz. "
tokenizer = CharTokenizer(text=text)


def test_decode_encode():
    assert s == tokenizer.decode(tokenizer.encode(s))


def test_vocab_size():
    assert tokenizer.vocab_size == len(set(text)) + 1


def test_unknown_str():
    assert tokenizer.encode(unknow_str) == [tokenizer.vocab_size]
    assert tokenizer.decode([tokenizer.vocab_size]) == "\ufffd"
