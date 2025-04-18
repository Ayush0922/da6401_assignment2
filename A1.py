import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
from torch import optim

class CustomCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.image_dim = cfg['img_size']
        self.total_compute = 0
        self.input_channels = cfg['input_channel']
        self.output_classes = cfg['output_neuron']

        self.filters = self._get_filters(cfg['filter_organisation'], cfg['no_of_filters'])
        self.activations = self._get_activations(cfg['activation'])

        self.conv_blocks = nn.ModuleList()
        self._create_conv_blocks()

        final_dim = self.image_dim * self.image_dim * self.filters[-1]
        self.flatten_layer = nn.Flatten()
        self.hidden_layer = nn.Linear(final_dim, cfg['dense_layer_size'])
        self.output_layer = nn.Linear(cfg['dense_layer_size'], self.output_classes)

        self.total_compute += final_dim * cfg['dense_layer_size'] + cfg['dense_layer_size']
        self.total_compute += cfg['dense_layer_size'] * self.output_classes + self.output_classes
        print(f"Final Dense Computation: {self.total_compute}")

 def _get_filters(self, mode, base):
        match mode:
            case "same":
                return [base] * 5
            case "double":
                return [base * (2 ** i) for i in range(5)]
            case "half":
                return [base // (2 ** i) for i in range(5)]

    def _get_activations(self, kind):
        match kind:
            case 'ReLU': return [nn.ReLU()] * 5
            case 'GELU': return [nn.GELU()] * 5
            case 'SiLU': return [nn.SiLU()] * 5
            case 'Mish': return [nn.Mish()] * 5

    def _conv_out_size(self, size, kernel, stride, padding):
        return (size - kernel + 2 * padding) // stride + 1

    def _pool_out_size(self, size, kernel, stride, padding, dilation=1):
        return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

    def _create_conv_blocks(self):
        idx = 0
        in_c = self.input_channels

        while idx < 5:
            out_c = self.filters[idx]
            conv = nn.Conv2d(in_channels=in_c,
                             out_channels=out_c,
                             kernel_size=self.cfg['filter_size'][idx],
                             stride=self.cfg['stride'][idx],
                             padding=self.cfg['padding'][idx])
            
            activation = self.activations[idx]
            pool = nn.MaxPool2d(kernel_size=self.cfg['pool_filter_size'][idx],
                                stride=self.cfg['pool_stride'][idx],
                                padding=self.cfg['pool_padding'][idx])
            
            self.conv_blocks.append(nn.Sequential(conv, activation, pool))

            # Computation
            new_dim = self._conv_out_size(self.image_dim,
                                          self.cfg['filter_size'][idx],
                                          self.cfg['stride'][idx],
                                          self.cfg['padding'][idx])

            compute = ((self.cfg['filter_size'][idx] ** 2) * in_c * new_dim ** 2 + 1) * out_c
            self.total_compute += compute
            print(f"Layer {idx+1} Computation: {compute}")

            new_dim = self._pool_out_size(new_dim,
                                          self.cfg['pool_filter_size'][idx],
                                          self.cfg['pool_stride'][idx],
                                          self.cfg['pool_padding'][idx])

            self.image_dim = new_dim
            in_c = out_c
            idx += 1

    def forward(self, x):
        for i, block in enumerate(self.conv_blocks, 1):
            x = block(x)
            print(f"After Layer {i}:", x.shape)

        x = self.flatten_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return F.softmax(x, dim=1)

def prepare_dataloaders(batch_size, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset_path = "inaturalist_12K"
    train_set = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform)
    test_set = datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform=transform)

    labels = train_set.classes
    train_subset, val_subset = random_split(train_set, [8000, 1999])

    return (labels,
            DataLoader(train_subset, batch_size=batch_size),
            DataLoader(val_subset, batch_size=batch_size),
            DataLoader(test_set, batch_size=batch_size))

# Config Setup
cfg = {
    'input_channel': 3,
    'output_neuron': 10,
    'filter_organisation': 'same',
    'no_of_filters': 8,
    'filter_size': [3]*5,
    'stride': [1]*5,
    'padding': [0]*5,
    'pool_filter_size': [3]*5,
    'pool_stride': [1]*5,
    'pool_padding': [0]*5,
    'activation': 'ReLU',
    'dense_layer_size': 16,
    'batch_size': 64,
    'img_size': 256,
}

# Model + Data
labels, train_dl, val_dl, test_dl = prepare_dataloaders(cfg['batch_size'], cfg['img_size'])
net = CustomCNN(cfg)
print(net)

loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(net.parameters(), lr=1e-4)

# Parameter Count
param_count = sum(p.numel() for p in net.parameters())
print(f"Total Parameters: {param_count}")

