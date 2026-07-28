# imagenet_lt.py
# -*- coding: utf-8 -*-
"""
ImageNet-LT loader built on top of imagenet.py.

Adds:
- lt_imb_factor: None (balanced) or a positive number (e.g., 10, 50, 100)
- lt_imb_type: "exp" (default)

When lt_imb_factor is set, the training split is sub-sampled to form
a long-tailed distribution with imbalance ratio IR = n_max / n_min.
"""

import os
import math
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_imagenet_dataset(root, split, transform):
    split_dir = os.path.join(root, split)
    if os.path.isdir(split_dir):
        return datasets.ImageFolder(root=split_dir, transform=transform)
    return datasets.ImageNet(root=root, split=split, transform=transform)


def _make_longtailed_indices(
    labels: np.ndarray,
    num_classes: int,
    lt_imb_factor: float,
    lt_imb_type: str = "exp",
    rng: np.random.Generator | None = None,
):
    assert lt_imb_factor is None or lt_imb_factor > 0

    if rng is None:
        rng = np.random.default_rng()

    N = labels.shape[0]
    classes = np.arange(num_classes)

    per_class_all = {c: np.where(labels == c)[0] for c in classes}
    per_class_counts = {c: len(per_class_all[c]) for c in classes}
    n_max = max(per_class_counts.values())

    if lt_imb_factor is None:
        all_indices = np.arange(N)
        rng.shuffle(all_indices)
        return all_indices.tolist()

    if lt_imb_type.lower() != "exp":
        raise ValueError("Only 'exp' lt_imb_type is supported.")

    C = float(num_classes)
    n_targets = {}

    for c in classes:
        ratio_exp = c / (C - 1.0) if C > 1 else 0.0
        n_c = n_max * (lt_imb_factor ** (-ratio_exp))
        n_c = max(1, int(round(n_c)))
        n_c = min(n_c, per_class_counts[c])
        n_targets[c] = n_c

    lt_indices = []
    for c in classes:
        idx_pool = per_class_all[c]
        if n_targets[c] >= len(idx_pool):
            chosen = idx_pool
        else:
            chosen = rng.choice(idx_pool, size=n_targets[c], replace=False)
        lt_indices.append(chosen)

    lt_indices = np.concatenate(lt_indices, axis=0)
    rng.shuffle(lt_indices)
    return lt_indices.tolist()


def get_train_valid_loader(batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=47,
                           pin_memory=True,
                           get_val_temp=0,
                           lt_imb_factor=None,
                           lt_imb_type="exp"):
    """
    ImageNet / ImageNet-LT train + validation loader.
    """

    assert 0.0 <= valid_size <= 1.0

    normalize = transforms.Normalize(
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD
    )

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = valid_transform

    data_dir = '../Datasets/ImageNet'

    train_dataset = _get_imagenet_dataset(
        root=data_dir, split='train', transform=train_transform
    )

    valid_dataset = _get_imagenet_dataset(
        root=data_dir, split='train', transform=valid_transform
    )

    # Extract labels
    if hasattr(train_dataset, "targets"):
        labels = np.array(train_dataset.targets)
    elif hasattr(train_dataset, "samples"):
        labels = np.array([s[1] for s in train_dataset.samples])
    else:
        raise RuntimeError("Cannot find labels in ImageNet dataset.")

    num_classes = len(train_dataset.classes)

    rng = np.random.default_rng(random_seed)

    lt_indices = _make_longtailed_indices(
        labels=labels,
        num_classes=num_classes,
        lt_imb_factor=lt_imb_factor,
        lt_imb_type=lt_imb_type,
        rng=rng,
    )

    num_train = len(lt_indices)
    split = int(math.floor(valid_size * num_train))

    if shuffle:
        rng.shuffle(lt_indices)

    valid_idx = lt_indices[:split]
    train_idx = lt_indices[split:]

    valid_temp_loader = None

    if get_val_temp and get_val_temp > 0:
        temp_split = int(math.floor(get_val_temp * len(valid_idx)))
        valid_idx, valid_temp_idx = valid_idx[temp_split:], valid_idx[:temp_split]

        valid_temp_dataset = _get_imagenet_dataset(
            root=data_dir, split='train', transform=valid_transform
        )

        valid_temp_sampler = SubsetRandomSampler(valid_temp_idx)
        valid_temp_loader = torch.utils.data.DataLoader(
            valid_temp_dataset,
            batch_size=batch_size,
            sampler=valid_temp_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if valid_temp_loader is not None:
        return train_loader, valid_loader, valid_temp_loader
    else:
        return train_loader, valid_loader


def get_test_loader(batch_size,
                    shuffle=True,
                    num_workers=23,
                    pin_memory=True):
    """
    Balanced ImageNet validation set as test set.
    """

    normalize = transforms.Normalize(
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD
    )

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = '../Datasets/ImageNet'

    dataset = _get_imagenet_dataset(
        root=data_dir,
        split='val',
        transform=transform
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )