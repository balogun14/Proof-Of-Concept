from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Dataset of images.

    Parameters
    ----------
    image_dir: Path
        Directory to the images on the disk.
    transform: Callable
        Transform function.
    """

    def __init__(self, image_dir: Path, transform: Callable):
        self.files = list(image_dir.glob("*.jpg"))
        self.transform = transform

    def __len__(self) -> int:
        """
        Number of elements in the dataset.
        """
        return len(self.files)

    def __getitem__(self, i: int) -> np.ndarray:
        """
        Get datapoint.

        Parameters
        ----------
        i: int
            Index of the image to get.

        Returns
        -------
        _: np.ndarray
            Image data.
        """
        image = Image.open(self.files[i])
        return self.transform(image)
