import torch
from torch import Tensor

# Comparison Functions

def cosine_similarity(t1: Tensor, t2: Tensor) -> float:
    t1 = torch.nn.functional.normalize(t1)
    t2 = torch.nn.functional.normalize(t2)
    return torch.nn.functional.cosine_similarity(t1, t2).item()

# Singular Functions

def norm(t1: Tensor) -> float:
    return torch.linalg.vector_norm(t1).item()