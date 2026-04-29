import numpy as np
import pytest
import torch
import torch.nn as nn

from poc.model import AutoEncoder


@pytest.mark.parametrize(
    "combine_spatial",
    [True, False],
)
def test_autoencoder_shape(combine_spatial: bool):
    """
    Test autoencoder output shape.

    Parameters
    ----------
    combine_spatial: bool
        Autoencoder that combines spatial dimensions (or not).
    """
    device = "cpu"
    dtype = torch.float32

    autoencoder = AutoEncoder(
        combine_spatial=combine_spatial, final_activation=nn.Sigmoid()
    ).to(device=device, dtype=dtype)
    x = torch.tensor(np.random.rand(1, 3, 224, 224), device=device, dtype=dtype)

    with torch.inference_mode():
        y = autoencoder(x)

    assert y.shape == x.shape


@pytest.mark.parametrize(
    "combine_spatial",
    [True, False],
)
def test_autoencoder_activation(combine_spatial: bool):
    """
    Test autoencoder output range with Sigmoid activation.

    Parameters
    ----------
    combine_spatial: bool
        Autoencoder that combines spatial dimensions (or not).
    """
    device = "cpu"
    dtype = torch.float32

    autoencoder = AutoEncoder(
        combine_spatial=combine_spatial, final_activation=nn.Sigmoid()
    ).to(device=device, dtype=dtype)
    x = torch.tensor(np.random.rand(1, 3, 224, 224), device=device, dtype=dtype)

    with torch.inference_mode():
        y = autoencoder(x)

    assert torch.all(y >= 0) and torch.all(y <= 1)


@pytest.mark.parametrize(
    "combine_spatial",
    [True, False],
)
def test_autoencoder_no_activation(combine_spatial: bool):
    """
    Test autoencoder works without final activation.

    Parameters
    ----------
    combine_spatial: bool
        Autoencoder that combines spatial dimensions (or not).
    """
    device = "cpu"
    dtype = torch.float32

    autoencoder = AutoEncoder(
        combine_spatial=combine_spatial, final_activation=None
    ).to(device=device, dtype=dtype)
    x = torch.tensor(np.random.rand(1, 3, 224, 224), device=device, dtype=dtype)

    with torch.inference_mode():
        y = autoencoder(x)

    assert y.shape == x.shape
