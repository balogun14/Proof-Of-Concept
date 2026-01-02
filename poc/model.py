import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    AutoEncoder.
    """

    def __init__(self, combine_spatial: bool):
        super().__init__()

        # Encoder: 3 -> 64 → 32
        if combine_spatial:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, stride=2, padding=1),
                nn.ReLU(),
            )

        else:
            self.encoder = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),
            )

        # Decoder: 32 → 16 → 3
        if combine_spatial:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 3, 2, stride=2, padding=0),
            )

        else:
            self.decoder = nn.Sequential(
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                ),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.ReLU(),
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                ),
                nn.Conv2d(16, 3, 3, padding=1),
            )

        self.final = nn.Sigmoid()

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
        x = self.encoder(x)
        x = self.decoder(x)
        return self.final(x)
