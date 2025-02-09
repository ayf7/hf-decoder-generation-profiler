import torch
from torch import Tensor

# Comparison Functions

def _cosine_similarity(t1: Tensor, t2: Tensor):
    t1 = torch.nn.functional.normalize(t1)
    t2 = torch.nn.functional.normalize(t2)
    return torch.clamp(torch.nn.functional.cosine_similarity(t1, t2), -1.0, 1.0)


def cosine_similarity(t1: Tensor, t2: Tensor) -> float:
    return _cosine_similarity(t1, t2).item()


def rad_angle(t1: Tensor, t2: Tensor) -> float:
    sim = _cosine_similarity(t1, t2)
    return torch.acos(sim).item()


def degree_angle(t1: Tensor, t2: Tensor) -> float:
    sim = _cosine_similarity(t1, t2)
    return torch.rad2deg(torch.acos(sim)).item()

# Singular Functions

def norm(t1: Tensor) -> float:
    return torch.linalg.vector_norm(t1).item()