import os
import random

import numpy as np
import torch
from torchvision import datasets, transforms


def seed_worker(worker_id):
    # See https://pytorch.org/docs/stable/notes/randomness.html
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# def FMNIST_loader(batch_size, train, generator=None, workers=16):
#     return torch.utils.data.DataLoader(
#         datasets.FashionMNIST(
#             "data/FashionMNIST",
#             train=train,
#             download=True,
#             transform=transforms.Compose(
#                 [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
#             ),
#         ),
#         batch_size=batch_size,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=workers,
#         worker_init_fn=seed_worker if generator is not None else None,
#         generator=generator,
#         persistent_workers=True,
#     )


def FMNIST_loader(data_path, batch_size, train, generator=None, workers=4, weighted_sampler=False):
    dataset = datasets.FashionMNIST(
        data_path,
        train=train,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        ),
    )
    if weighted_sampler:
        class_weight = 0.9
        num_classes = 10
        fill_weight = (1 - class_weight) / num_classes
        weights = [class_weight if l ==
                   0 else fill_weight for (_, l) in dataset]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            len(dataset),
            replacement=True,
            generator=generator,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler if weighted_sampler else None,
        pin_memory=True,
        num_workers=workers,
        worker_init_fn=seed_worker if generator is not None else None,
        generator=generator,
        persistent_workers=True,
    )


def ImageNet_loader(data_path, batch_size, train, generator=None, workers=4):
    data_dir = os.path.join(
        data_path, 'train') if train else os.path.join(data_path, 'val')
    # Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if train:
        trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    dataset = datasets.ImageFolder(data_dir, trans)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker if generator is not None else None,
        generator=generator,
        persistent_workers=True,
    )
