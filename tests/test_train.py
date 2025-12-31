from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from poc.config import SMALL_CONFIG
from poc.model import AutoEncoder
from poc.train import train


def test_train_autoencoder():
    """
    Test autoencoder training.
    """
    device = "mps"
    dtype = torch.bfloat16
    batch_size = 16

    config = SMALL_CONFIG
    config.batch_size = batch_size
    config.device = device
    config.dtype = dtype

    base_dir = Path(__file__).parent.parent

    model = AutoEncoder().to(device=device, dtype=dtype)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    x = torch.tensor(np.random.rand(100, 3, 224, 224)).to(
        device=device, dtype=dtype
    )
    dataset = TensorDataset(x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train(
        output_dir=base_dir / "data" / "out" / "test_train",
        loader=dataloader,
        model=model,
        optimizer=optimizer,
        config=config,
    )
    y01 = model(x[0, ...])

    weights = torch.load(
        base_dir / "data" / "out" / "test_train" / "model.pt",
        map_location=device,
    )
    model = AutoEncoder().to(device=device, dtype=dtype)
    model.load_state_dict(weights)
    y02 = model(x[0, ...])

    assert torch.equal(y01, y02)
