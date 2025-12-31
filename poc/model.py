import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    AutoEncoder.
    """

    def __init__(self):
        super().__init__()

        # Encoder: 128 → 64 → 32
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 128 → 64
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 64 → 32
            nn.ReLU(),
        )

        # Decoder: 32 → 64 → 128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        _: torch.Tensor
            Output tensor.
        """
        return self.decoder(self.encoder(x))
