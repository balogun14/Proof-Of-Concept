import numpy as np
import torch

from poc.model import AutoEncoder


def test_autoencoder():
    """
    Test autoencoder output shape.
    """
    device = "mps"
    dtype = torch.bfloat16

    autoencoder = AutoEncoder().to(device=device, dtype=dtype)
    x = torch.tensor(np.random.rand(1, 3, 224, 224)).to(
        device=device, dtype=dtype
    )

    with torch.inference_mode():
        y = autoencoder(x)

    assert y.shape == x.shape
