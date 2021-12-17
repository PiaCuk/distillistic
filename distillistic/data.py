import random

import numpy as np
import torch
from torchvision import datasets, transforms


def seed_worker(worker_id):
    # See https://pytorch.org/docs/stable/notes/randomness.html
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def FMNIST_loader(batch_size, train, generator=None, workers=16):
    return torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "data/FashionMNIST",
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=workers,
        worker_init_fn=seed_worker if generator is not None else None,
        generator=generator,
        persistent_workers=True,
    )


def FMNIST_weighted_loader(batch_size, train, generator=None, workers=16):
    dataset = datasets.FashionMNIST(
        "data/FashionMNIST",
        train=train,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        ),
    )
    class_weight = 0.9
    num_classes = 10
    fill_weight = (1 - class_weight) / num_classes
    weights = [class_weight if l == 0 else fill_weight for (_, l) in dataset]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights,
        len(dataset),
        replacement=True,
        generator=generator,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=workers,
        worker_init_fn=seed_worker if generator is not None else None,
        generator=generator,
        persistent_workers=True,
    )
