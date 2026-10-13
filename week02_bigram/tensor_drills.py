import torch

zero_mat = torch.zeros([3, 4], dtype=torch.float32, device="cpu")  # shape(3, 4)
one_mat = torch.ones([3, 4], dtype=torch.float32, device="mps")  # shape(3, 4)
rand_mat = torch.rand([3, 4], dtype=torch.float16, device="cpu")  # shape(3, 4)
range_mat = torch.arange(0, 5, 1, dtype=torch.float32, device="mps")  # shape(5,)

x = torch.randn(2, 4, 8)  # shape(2, 4, 8)
y = x.transpose(1, 2)  # shape(2, 8, 4)
z = y.reshape(2, -1)  # shape(2, 32)
# w = y.view(2, -1)  # 这行会报错——这就是今天要观察的东西
w = y.contiguous().view(2, -1)  # shape(2, 32)
p = torch.permute(w, (1, 0))  # shape(32, 2)
print(f"p.shape is: {p.shape}\n")

B = 5
T = 2
C = 6
H = 4
shape_one = (B, T, 1)  # shape(5, 2, 1)
shape_two = (1, 1, C)  # shape(1, 1, 6)
shape_mask = (T, T)  # shape(2, 2)
shape_batch = (B, H, T, T)
mat_one = torch.zeros(shape_one)  # shape(5, 2, 1)
mat_two = torch.ones(shape_two)  # shape(1, 1, 6)
mat_three = mat_one + mat_two  # shape(5, 2, 6)
print(f"mat_three shape is: {mat_three.shape}, mat_three is: {mat_three}\n")
mask = torch.full(shape_mask, -float("inf"))  # shape(2, 2)
batch = torch.rand(shape_batch)  # shape(5, 4, 2, 2)
batch_mask = batch + mask  # shape(5, 4, 2, 2)
