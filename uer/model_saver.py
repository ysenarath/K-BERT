# -*- encoding:utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Model(Protocol):
    module: torch.nn.Module


def save_model(model: Model | torch.nn.Module, model_path: str | Path):
    if isinstance(model, Model):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
