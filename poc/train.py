from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from poc.config import TrainConfig
from poc.dataset import ImageDataset
from poc.model import AutoEncoder


def backward(
    x: torch.Tensor,
    model: nn.Module,
    config: TrainConfig,
):
    """
    Backward the diffusion branch.

    Parameters
    ----------
    x: torch.Tensor
        Input tensor.
    model: nn.Module
        Model.
    config: TrainConfig
        Train config.

    Returns
    -------
    grads: {str: torch.Tensor}
        Dictionary of gradients.
    """
    y = model(x)
    loss = MSELoss()(y, x)
    loss.backward()

    if config.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=config.max_grad_norm,
        )


def train(
    output_dir: Path,
    loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
):
    """
    Train foundational model.

    Parameters
    ----------
    output_dir: Path
        Path for saving model checkpoints.
    loader: DataLoader
        Image loader.
    model: Embedder
        Model to train.
    optimizer: torch.optim.Optimizer
        Optimizer.
    config: TrainConfig
        Train config.
    """
    model.train()

    for _ in range(config.nb_epochs):
        for x in tqdm(loader):
            if len(x) < config.batch_size:
                continue

            x = x.to(device=config.device, dtype=config.dtype)
            optimizer.zero_grad()
            backward(x=x, model=model, config=config)
            optimizer.step()

    logger.info("Saving models.")
    torch.save(model.state_dict(), output_dir / "model.pt")
    logger.info(f"Models saved at {output_dir / 'model.pt'}.")


def main():
    """
    Entrypoint for training foundational model.
    """
    from poc.config import SMALL_CONFIG

    config = SMALL_CONFIG

    base_dir = Path(__file__).parent.parent
    image_dir = base_dir / "data" / "in" / "19"

    train_transform = transforms.Compose([
        # transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.RandomResizedCrop(
            size=224, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageDataset(
        image_dir,
        transform=train_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    logger.info(f"Number of elements: {len(dataset)}.")
    logger.info(f"Batches per epoch: {config.batch_size}.")
    logger.info(
        f"Total training steps: {len(dataset) // config.batch_size * config.nb_epochs}."
    )

    logger.info("Building model.")
    model = AutoEncoder().to(device=config.device, dtype=config.dtype)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    logger.info("Training model.")
    train(
        output_dir=base_dir / "data" / "out",
        loader=loader,
        model=model,
        optimizer=optimizer,
        config=config,
    )


if __name__ == "__main__":
    main()
