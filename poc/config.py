import torch
from pydantic import BaseModel


class TrainConfig(BaseModel):
    device: str
    dtype: torch.dtype
    nb_epochs: int
    batch_size: int
    lr: float = 1e-5
    max_grad_norm: float | None = None
    project_name: str = "poc-autoencoder"
    use_wandb: bool = False


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


SMALL_CONFIG = TrainConfig(
    device=get_device(),
    dtype=torch.float32,
    nb_epochs=3,
    batch_size=12,  # 12 64
    max_grad_norm=1.0,
    project_name="poc-autoencoder",
    use_wandb=True,
)
