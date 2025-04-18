import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from matplotlib import patches

# Device and worker config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = min(os.cpu_count(), 8)

# Activation mapping
def get_activation(name):
    return {"ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
        "Mish": nn.Mish
    }.get(name, nn.ReLU)

# Transforms
make_resize = lambda sz: transforms.Resize((sz, sz))

def define_transforms(size, augment):
    if augment == 'Yes':
        t_train = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])
    else:
        t_train = transforms.Compose([make_resize(size), transforms.ToTensor()])
    t_test = transforms.Compose([make_resize(size), transforms.ToTensor()])
    return t_train, t_test

# Dataloaders

def create_dataloaders(train_path, test_path, batch_size, img_size, augment):
    t_train, t_test = define_transforms(img_size, augment)
    all_data = datasets.ImageFolder(train_path, transform=t_train)
    test_data = datasets.ImageFolder(test_path, transform=t_test)

    y_labels = np.array([s[1] for s in all_data.samples])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in splitter.split(np.zeros(len(y_labels)), y_labels):
        train_subset = Subset(all_data, train_idx)
        val_subset = Subset(datasets.ImageFolder(train_path, transform=t_test), val_idx)

    def load(subset, shuffle=False):
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=WORKERS, pin_memory=True)

    return load(train_subset, True), load(val_subset), load(test_data), len(all_data.classes), all_data.classes

