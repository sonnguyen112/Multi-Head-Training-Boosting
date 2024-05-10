import torch
from torch import nn

m = nn.AdaptiveAvgPool2d(1)
input = torch.randn(1, 64, 8, 9)
b, c, _, _ = input.size()
output = m(input).view(b, c)
print(output.shape)
print(output)