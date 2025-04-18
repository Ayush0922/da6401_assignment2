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

# Training pipeline
def train_loop(model, loader, criterion, optimizer):
    model.train()
    batch_idx = 0
    while batch_idx < len(loader):
        for inputs, targets in [loader[batch_idx]]:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
        batch_idx += 1

def validate_loop(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        index = 0
        while index < len(loader):
            for inputs, labels in [loader[index]]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            index += 1
    return correct / total

# Main training function
def experiment(cfg=None):
    with wandb.init(config=cfg):
        params = wandb.config
        params = namedtuple("Params", params.keys())(*params.values())

        train_loader, val_loader, classes = fetch_data(
            DATA_PATH, params.batch_size, params.img_size, params.data_augmentaion == 'Yes')

        setattr(params, 'num_classes', classes)

        model = Net(params).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=params.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        epoch_counter = 0
        while epoch_counter < params.epochs:
            train_loop(model, train_loader, loss_fn, opt)
            acc = validate_loop(model, val_loader)
            wandb.log({"accuracy": acc})
            epoch_counter += 1

# Sweep and config
SWEEP_CFG = {
    'name': 'vision_hyper_sweep',
    'method': 'bayes',
    'metric': {'goal': 'maximize', 'name': 'accuracy'},
    'parameters': {
        'epochs': {'values': [5,10,15,25]},
        'batch_size': {'values': [32, 64, 128]},
        'learning_rate': {'values': [0.0001,0.00001,0.000001]},
        'dense_size': {'values': [256]},
        'filter_size': {'values': [[7, 5, 5, 3, 3], [11, 7, 5, 3, 3], [3, 3, 3, 3, 3]]},
        'activation': {'values': ['LeakyReLU', 'ReLU', 'GELU', 'SiLU', 'Mish']},
        'filter_organisation': {'values': ['same', 'alt']},
        'no_of_filters': {'values': [64]},
        'data_augmentaion': {'values': ['No', 'Yes']},
        'batch_normalization': {'values': ['Yes', 'No']},
        'dropout': {'values': [0.2, 0.3]},
        'img_size': {'values': [224, 256]},
        'optimizer': {'values': ['adam']}
    }
}

DATA_PATH = "/kaggle/input/nature-922/inaturalist_12K/train"
sweep_id = wandb.sweep(SWEEP_CFG, project="iNaturalist-CNN-2")
wandb.agent(sweep_id, function=experiment)
