from typing import Any

import torch
from pydantic import BaseModel


class TrainConfig(BaseModel):
    device: str
    dtype: Any
    nb_epochs: int
    batch_size: int
    lr: float = 1e-5
    max_grad_norm: float | None = None


SMALL_CONFIG = TrainConfig(
    device="mps",
    dtype=torch.float32,
    nb_epochs=1,
    batch_size=12,  # 12 64
    max_grad_norm=1.0,
)
