import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from poc.config import SMALL_CONFIG
from poc.model import AutoEncoder
from poc.train import train


@pytest.mark.parametrize(
    "combine_spatial",
    [True, False],
)
def test_train_autoencoder(combine_spatial: bool):
    """
    Test autoencoder training.

    Parameters
    ----------
    combine_spatial: bool
        Autoencoder that combines spatial dimensions (or not).
    """
    device = "mps"
    dtype = torch.bfloat16
    batch_size = 16

    config = SMALL_CONFIG
    config.batch_size = batch_size
    config.device = device
    config.dtype = dtype

    base_dir = Path(__file__).parent.parent
    shutil.rmtree(base_dir / "data" / "out" / "test_train", ignore_errors=True)

    model = AutoEncoder(
        combine_spatial=combine_spatial, final_activation=nn.Sigmoid()
    ).to(device=device, dtype=dtype)
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
    y01 = model(x[0, ...].unsqueeze(0))

    weights = torch.load(
        base_dir / "data" / "out" / "test_train" / "model.pt",
        map_location=device,
    )
    model = AutoEncoder(
        combine_spatial=combine_spatial, final_activation=nn.Sigmoid()
    ).to(device=device, dtype=dtype)
    model.load_state_dict(weights)
    y02 = model(x[0, ...].unsqueeze(0))

    assert torch.equal(y01, y02)
