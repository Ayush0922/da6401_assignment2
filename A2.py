import os
import random
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from enum import Enum
from collections import namedtuple

# Device Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKER_THREADS = min(os.cpu_count(), 8)

# Enum for activation selection
class ActivationType(Enum):
    RELU = "ReLU"
    LEAKY_RELU = "LeakyReLU"
    GELU = "GELU"
    SILU = "SiLU"
    MISH = "Mish"

ACTIVATION_MAP = {
    ActivationType.RELU: nn.ReLU,
    ActivationType.LEAKY_RELU: nn.LeakyReLU,
    ActivationType.GELU: nn.GELU,
    ActivationType.SILU: nn.SiLU,
    ActivationType.MISH: nn.Mish
}

# Data transformation logic
def build_transforms(img_size, augment=True):
    base = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    if augment:
        aug = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ]
        return transforms.Compose(aug + [transforms.ToTensor()]), transforms.Compose(base)
    return transforms.Compose(base), transforms.Compose(base)

# Stratified dataset split
def stratify(dataset, labels, split_ratio=0.2):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=42)
    for train_idx, val_idx in splitter.split(np.zeros(len(labels)), labels):
        return train_idx, val_idx

# Data preparation and loading
def fetch_data(path, batch, img_sz, augment):
    train_tf, val_tf = build_transforms(img_sz, augment)
    raw = datasets.ImageFolder(path, transform=train_tf)
    label_arr = np.array([s[1] for s in raw.samples])
    tr_idx, va_idx = stratify(raw, label_arr)
    val_data = datasets.ImageFolder(path, transform=val_tf)

    return (
        DataLoader(Subset(raw, tr_idx), batch_size=batch, shuffle=True, num_workers=WORKER_THREADS, pin_memory=True),
        DataLoader(Subset(val_data, va_idx), batch_size=batch, shuffle=False, num_workers=WORKER_THREADS, pin_memory=True),
        len(raw.classes)
    )

# Network definition
class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = self._build_conv_block(config)
        self.flattened_size = self._get_flattened_size(config.img_size)
        self.dense = self._build_fc(config)

    def _build_conv_block(self, cfg):
        filters = [cfg.no_of_filters * (2 if cfg.filter_organisation == 'alt' and i % 2 else 1)
                   for i in range(len(cfg.filter_size))]
        act = ACTIVATION_MAP[ActivationType(cfg.activation)]
        sequence = nn.Sequential()
        in_ch = 3

        i = 0
        while i < len(cfg.filter_size):
            out_ch = filters[i]
            sequence.append(nn.Conv2d(in_ch, out_ch, kernel_size=cfg.filter_size[i], padding=cfg.filter_size[i] // 2))
            if cfg.batch_normalization == 'Yes':
                sequence.append(nn.BatchNorm2d(out_ch))
            sequence.append(act())
            sequence.append(nn.MaxPool2d(kernel_size=2))
            if cfg.dropout > 0:
                sequence.append(nn.Dropout2d(cfg.dropout))
            in_ch = out_ch
            i += 1
        return sequence

    def _get_flattened_size(self, img_sz):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_sz, img_sz)
            out = self.conv(dummy)
            return out.view(1, -1).size(1)

    def _build_fc(self, cfg):
        act = ACTIVATION_MAP[ActivationType(cfg.activation)]
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, cfg.dense_size),
            act(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dense_size, cfg.num_classes)
        )

    def forward(self, x):
        return self.dense(self.conv(x))

