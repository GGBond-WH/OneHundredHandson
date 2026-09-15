import torch


def debug_drill():
    x = torch.randn(2, 4, 8)
    y = x.transpose(1, 2)
    z = y.reshape(2, -1)
    k = y.view(2, 8, 4)
    w = y.view(2, -1)


debug_drill()
