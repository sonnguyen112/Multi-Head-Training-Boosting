import torch

a = torch.tensor([1, 2]).to(dtype=torch.float)
b = torch.tensor([1, 2]).to(dtype=torch.float)

cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
print(cosine(a, b))
print(1 - cosine(a,b))