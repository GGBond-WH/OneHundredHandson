from week01_text_pipeline.text_pipeline.bpe import BasicTokenizer, get_stats, merge

ids = [1, 2, 2, 1, 2]
pair = (1, 2)
new_id = 9
tokenizer = BasicTokenizer()
text = "aaabdaaabac"
PARAGRAPH = (
    "The cache stores the keys and the values. During decoding, the model reads the cache "
    "instead of recomputing the keys and the values for every token. The longer the context, "
    "the larger the cache; the larger the cache, the more memory the model needs. "
    "A static cache allocates the memory once. A dynamic cache grows with the context. "
    "A sliding cache keeps only the most recent tokens. Testing the cache means checking that "
    "the cached logits match the full logits, token by token."
)
UNICODE = "KV cache 把每一层的 K 和 V 存起来，decode 时每步只算一个新 token。缓存越长，显存越大 🚀🚀"
vocab_size = 280


def test_get_stats():
    assert get_stats(ids) == {(1, 2): 2, (2, 2): 1, (2, 1): 1}


def test_merge():
    assert merge(ids, pair, new_id) == [9, 2, 9]


def test_merge_len():
    tokenizer.train(PARAGRAPH, vocab_size)
    assert len(tokenizer.merges) == vocab_size - 256


def test_encode_decode():
    tokenizer = BasicTokenizer()
    tokenizer.train(UNICODE, 300)
    assert tokenizer.decode(tokenizer.encode(UNICODE)) == UNICODE


def test_two_train():
    tokenizer_first = BasicTokenizer()
    tokenizer_first.train(PARAGRAPH, 300)
    tokenizer_second = BasicTokenizer()
    tokenizer_second.train(PARAGRAPH, 300)
    assert tokenizer_first.merges == tokenizer_second.merges
    assert len(tokenizer_second.vocab) == 300
    assert len(tokenizer_first.merges) == 44
