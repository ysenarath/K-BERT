# -*- encoding:utf-8 -*-
import torch


def flip(x: torch.Tensor, dim: int) -> torch.Tensor:
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]
