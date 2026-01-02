import numpy as np
import pytest
import torch
import torch.nn as nn

from poc.model import AutoEncoder


@pytest.mark.parametrize(
    "combine_spatial",
    [True, False],
)
def test_autoencoder(combine_spatial: bool):
    """
    Test autoencoder output shape.

    Parameters
    ----------
    combine_spatial: bool
        Autoencoder that combines spatial dimensions (or not).
    """
    device = "mps"
    dtype = torch.bfloat16

    autoencoder = AutoEncoder(
        combine_spatial=combine_spatial, final_activation=nn.Sigmoid()
    ).to(device=device, dtype=dtype)
    x = torch.tensor(np.random.rand(1, 3, 224, 224)).to(
        device=device, dtype=dtype
    )

    with torch.inference_mode():
        y = autoencoder(x)

    assert y.shape == x.shape
