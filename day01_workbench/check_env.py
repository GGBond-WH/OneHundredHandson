import torch

print(
    f"torch version: {torch.__version__}, mps is available: {torch.backends.mps.is_available()}\n"
)
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
res = torch.matmul(torch.randn(3, 3, device=device), torch.randn(3, 3, device=device))
print(f"res is: {res}\n")
