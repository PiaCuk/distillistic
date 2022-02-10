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


def FMNIST_loader(data_path, batch_size, train, generator=None, workers=4, weighted_sampler=False):
    normalize = transforms.Normalize((0.2860,), (0.3530,))
    if train:
        trans = transforms.Compose([
            transforms.RandAugment(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])

    dataset = datasets.FashionMNIST(
        data_path,
        train=train,
        download=True,
        transform=trans,
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


IMAGENET_SIZE = 224


def _ImageNet_loader(data_path, batch_size, train, generator=None, workers=4, downscale=1):
    data_dir = os.path.join(
        data_path, 'train') if train else os.path.join(data_path, 'val')
    # Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if train:
        trans = [
            transforms.RandomResizedCrop(IMAGENET_SIZE),
            # transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        trans = [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGENET_SIZE),
            transforms.ToTensor(),
            normalize,
        ]

    if (downscale > 1) and ((downscale % 2) == 0):
        target_size = int(IMAGENET_SIZE / downscale)
        print(f"Downscaling images to {target_size}.")
        trans.append(transforms.Resize(target_size))
    else:
        print("Only downscale by multiples of 2. Not downscaling.")

    dataset = datasets.ImageFolder(data_dir, transforms.Compose(trans))

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


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
RES_TUPLE = (IMAGENET_SIZE, IMAGENET_SIZE)
DEFAULT_CROP_RATIO = IMAGENET_SIZE / 256


def FFCV_ImageNet_loader(data_path, batch_size, device, train, workers=4, in_memory=False, use_amp=True):
    """
    :param in_memory (bool): Does the dataset fit in memory?
    :param use_amp (bool): True to use Automated Mixed Precision
    """
    from ffcv.fields.basics import IntDecoder
    from ffcv.fields.rgb_image import (CenterCropRGBImageDecoder,
                                    RandomResizedCropRGBImageDecoder)
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import (NormalizeImage, RandomHorizontalFlip, Squeeze,
                                ToDevice, ToTensor, ToTorchImage)
    
    img_type = np.float16 if use_amp else np.float32

    if train:
        decoder = RandomResizedCropRGBImageDecoder(RES_TUPLE)
        image_pipeline = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, img_type)
        ]
    else:
        decoder = CenterCropRGBImageDecoder(
            RES_TUPLE, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            decoder,
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, img_type)
        ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]

    order = OrderOption.QUASI_RANDOM if train else OrderOption.SEQUENTIAL
    loader = Loader(data_path,
                    batch_size=batch_size,
                    num_workers=workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=train,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=False)

    return loader


def ImageNet_loader(data_path, batch_size, device, train, generator=None, workers=4, use_amp=False, use_ffcv=False, downscale=1):
    """
    :param data_path (str): Path to data
    :param batch_size (int): Batch size for DataLoader
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param train (bool): True for training set, False for validation set
    :param generator (torch.Generator): Generator for random numbers in DataLoader
    :param workers (int): Number of workers for DataLoader
    :param use_amp (bool): True to use Automatic Mixed Precision. Only used when use_ffcv=True
    :param use_ffcv (bool): True to load data with FFCV
    :param downscale (int): Downscaling factor. 1 for no downscaling
    """
    print(f"Load data with FFCV is set to {use_ffcv}.")
    if use_ffcv:
        ffcv_name = "train_500_0.50_90.ffcv" if train else "val_500_0.50_90.ffcv"
        ffcv_path = os.path.join(data_path, ffcv_name)
        return FFCV_ImageNet_loader(ffcv_path, batch_size, device, train, workers, use_amp=use_amp)
    else:
        return _ImageNet_loader(data_path, batch_size, train, generator, workers, downscale)