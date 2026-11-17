import math
import time

import attn_matrix
import torch
from attn_loop import com_loop
from torch.nn import functional as F

"""
loop vs matrix vs sdpa, 比较运行时间
1. 构造输入: [B, T, C], 测试不同的T, 分别取64, 256, 1024
2. 因为提到了sdpa, 所以想单独比较计算模块的计时, 就要考虑QKV投影部分不计算时间
"""
B, C, D, H = 2, 64, 64, 1
devices = [torch.device("cpu")]
if torch.backends.mps.is_available():
    devices.append(torch.device("mps"))
for device in devices:
    for T in (64, 256, 1024):
        x = torch.randn(B, T, C, device=device)
        wq = torch.randn(C, D, device=device) / math.sqrt(C)
        wk = torch.randn(C, D, device=device) / math.sqrt(C)
        wv = torch.randn(C, D, device=device) / math.sqrt(C)
        q = x @ wq
        k = x @ wk
        v = x @ wv
        q_trans = q.view(B, T, H, C // H).transpose(1, 2)
        k_trans = k.view(B, T, H, C // H).transpose(1, 2)
        v_trans = v.view(B, T, H, C // H).transpose(1, 2)
        model_matrix = attn_matrix.CausalSelfAttention(C, D, 1024).to(device)
        model_matrix.eval()
        with torch.no_grad():
            for _ in range(5):
                out_loop = com_loop(q, k, v)
                out_matrix = model_matrix.com_loop(q, k, v)
                out_sdpa = F.scaled_dot_product_attention(
                    q_trans, k_trans, v_trans, is_causal=True
                )
                out_sdpa = out_sdpa.squeeze(1)
                torch.testing.assert_close(out_loop, out_sdpa)

            T_sum = 0
            for _ in range(5):
                if device.type == "mps":
                    torch.mps.synchronize()
                start = time.perf_counter()
                out_loop = com_loop(q, k, v)
                if device.type == "mps":
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                T_sum += elapsed
            T_sum /= 5
            print(
                f"T: {T}, loop耗时: {T_sum * 1000:.3f} ms, device: {device.type}, dtype: {q.dtype}"
            )

            T_sum = 0
            for _ in range(5):
                if device.type == "mps":
                    torch.mps.synchronize()
                start = time.perf_counter()
                out_matrix = model_matrix.com_loop(q, k, v)
                if device.type == "mps":
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                T_sum += elapsed
            T_sum /= 5
            print(
                f"T: {T}, matrix平均耗时：{T_sum * 1000:.3f} ms, device: {device.type}, dtype: {q.dtype}"
            )

            T_sum = 0
            for _ in range(5):
                if device.type == "mps":
                    torch.mps.synchronize()
                start = time.perf_counter()
                out_sdpa = F.scaled_dot_product_attention(
                    q_trans, k_trans, v_trans, is_causal=True
                )
                if device.type == "mps":
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                T_sum += elapsed
            T_sum /= 5
            print(
                f"T: {T}, sdpa耗时：{T_sum * 1000:.3f} ms, device: {device.type}, dtype: {q.dtype}"
            )
