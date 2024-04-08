import torch

test_tensor = torch.tensor([1, 2, 3, 4])
a = test_tensor
a[1] = 0
print(test_tensor)