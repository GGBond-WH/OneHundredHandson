# 手写多头因果自注意力shape变化
B = 2, T = 8, C = 32, H = 4
x ---> [B, T, C]
|   # 一次性算出q、k、v，这里有一个线性层
qkv ---> [B, T, 3C]
|   # 从qkv中拆出q、k、v
q, k, v ---> [B, T, C]
|   # 按H个头开始切分最后一维，每个头维度是C // H = D
q ---> [B, T, H, D]
k ---> [B, T, H, D]
v ---> [B, T, H, D]
|   # 先转置第一维和第二维
q ---> [B, H, T, D]
k ---> [B, H, T, D]
v ---> [B, H, T, D]
|   # mask实现因果注意力，logits = q @ k ^ T
logits ---> [B, H, T, T]
|   # 经过softmax，并引入attn_dropout
weight ---> [B, H, T, T]
|   # 算出加权聚合输出，weight @ v
attn_output ---> [B, H, T, D]
|   # 转置第一维和第二维，将所有头拼起来
out ---> [B, T, C]
|   # 最后通过一个线性层，构建所有头之间的联系
out ---> [B, T, C]
|   # 使用residual dropout
out ---> [B, T, C]
