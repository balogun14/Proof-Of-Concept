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
    project_name: str | None = None
    use_wandb: bool = True


def get_device() -> str:
    """
    Get device to train on.

    Returns
    -------
    str
        Device to train on.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


SMALL_CONFIG = TrainConfig(
    device=get_device(),
    dtype=torch.float32,
    nb_epochs=1,
    batch_size=12,  # 12 64
    max_grad_norm=1.0,
    project_name="poc",
    use_wandb=False,
)
