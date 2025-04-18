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
