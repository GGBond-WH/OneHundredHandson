from pathlib import Path

import torch
from torch import optim

from week01_text_pipeline.text_pipeline.dataset import TextDataset
from week01_text_pipeline.text_pipeline.tokenizer import CharTokenizer
from week02_bigram.mini_lm.train import eval_loop, train_loop
from week04_gpt.model import GPT

n_layer = 4
n_head = 4
n_embd = 128
block_size = 64
batch = 32
lr = 3e-4
device = "mps"

text = Path("data/input.txt").read_text(encoding="utf-8")
tokenizer = CharTokenizer(text=text)
vocab_size = tokenizer.vocab_size
ids = tokenizer.encode(text)
split = int(len(ids) * 0.9)
train_ids = ids[:split]
val_ids = ids[split:]
train_dataset = TextDataset(train_ids, block_size)
val_dataset = TextDataset(val_ids, block_size)
model = GPT(vocab_size, n_embd, n_head, block_size, n_layer)
optimizer = optim.AdamW(model.parameters(), lr=lr)
train_loop(
    model,
    optimizer=optimizer,
    iter_num=2000,
    train_dataset=train_dataset,
    device=device,
    batch_size=batch,
)
eval_loop(model, iter_num=100, val_dataset=val_dataset, device=device, batch_size=batch)

prompt = "First Citizen:"
idx = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device=device)  # [B, T]
idx_new = idx
model.eval()
model.to(device=device)
with torch.no_grad():
    for _ in range(300):
        logits, _ = model(idx)
        prob = torch.softmax(logits[:, -1, :], dim=-1).squeeze(1)  # [B, V]
        next_id = torch.multinomial(prob, num_samples=1)  # [B, 1]
        idx_new = torch.cat((idx_new, next_id), dim=-1)  # [B, T+1]
        if len(idx_new[0]) >= block_size:
            breakpt = len(idx_new[0]) - block_size
            idx = idx_new[:, breakpt:]
        else:
            idx = idx_new
idx_new = idx_new[0].tolist()
text = tokenizer.decode(idx_new)
print(f"生成的字符串为：{text}\n")
