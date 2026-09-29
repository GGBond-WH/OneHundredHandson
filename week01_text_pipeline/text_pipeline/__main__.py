import argparse
from pathlib import Path

import requests

from .dataset import TextDataset, batch_iterator
from .io import load_text
from .tokenizer import CharTokenizer

parser = argparse.ArgumentParser(description="分词器训练及模型训练生成数据全流程")
parser.add_argument("--file", type=str, help="指定训练数据路径")
parser.add_argument("--block-size", type=int, help="指定分词器训练数据块大小")
parser.add_argument("--batch-size", type=int, help="指定分词器训练批大小")

args = parser.parse_args()

tiny_shakespeare = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
path = Path(args.file)

if not path.exists():
    with requests.get(url=tiny_shakespeare, stream=True, timeout=30) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

data_str = load_text(path)
tokenizer = CharTokenizer(data_str)
ids = tokenizer.encode(data_str)
dataset = TextDataset(ids, args.block_size)
batchs = batch_iterator(dataset=dataset, batch_size=args.batch_size)
first_batch = next(batchs)
print(f"第一批数据的ids是：{first_batch}\n")
