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

# CNN
class ConvNet(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.features = self._make_layers()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config['img_size'], config['img_size'])
            output = self.features(dummy)
            flat_size = output.view(1, -1).size(1)
        
        act_fn = get_activation(config['activation'])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, config['dense_size']),
            act_fn(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['dense_size'], num_classes)
        )

    def _make_layers(self):
        c = self.config
        in_ch = 3
        layers = []
        f_count = [c['no_of_filters'] if c['filter_organisation']=='same' else (c['no_of_filters'] if i%2==0 else c['no_of_filters']*2) for i in range(len(c['filter_size']))]
        act_fn = get_activation(c['activation'])

        i = 0
        while i < len(c['filter_size']):
            k = c['filter_size'][i]
            out_ch = f_count[i]
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(out_ch) if c['batch_normalization']=='Yes' else nn.Identity(),
                act_fn(),
                nn.MaxPool2d(2),
                nn.Dropout2d(c['dropout']) if c['dropout'] > 0 else nn.Identity()
            ])
            in_ch = out_ch
            i += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(self.features(x))

# Train

def execute_training(cfg, train_dl, val_dl, n_classes):
    model = ConvNet(cfg, n_classes).to(DEVICE)
    optim_ = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()
    best, weights = 0.0, None

    for ep in range(cfg['epochs']):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim_.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optim_.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                _, out = torch.max(pred, 1)
                correct += (out == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {ep+1}/{cfg['epochs']} - Val Acc: {acc:.4f}")
        if acc > best:
            best = acc
            weights = model.state_dict()

    model.load_state_dict(weights)
    return model

# Test Accuracy

def get_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct / total

# Visualizations

def visualize(model, test_loader, class_names, n_classes=10, samples=3):
    from matplotlib import pyplot as plt
    model.eval()
    sampled = {i: [] for i in range(n_classes)}
    preds, probs = {i: [] for i in range(n_classes)}, {i: [] for i in range(n_classes)}

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            o = model(x)
            _, p = torch.max(o, 1)
            ps = torch.softmax(o, dim=1)
            for img, gt, pd, prb in zip(x, y, p, ps):
                idx = gt.item()
                if len(sampled[idx]) < samples:
                    sampled[idx].append(img.cpu())
                    preds[idx].append(pd.item())
                    probs[idx].append(prb.cpu())
            if all(len(sampled[i]) == samples for i in sampled):
                break

    fig = plt.figure(figsize=(18, 25))
    fig.suptitle("Predictions", fontsize=24, y=0.98, fontweight='bold', color='#2e3440')
    gs = plt.GridSpec(n_classes, samples + 2, width_ratios=[1]*samples + [0.5, 1.5], hspace=0.4, wspace=0.1)

    for i in range(n_classes):
        for j in range(samples):
            ax = plt.subplot(gs[i, j])
            img = sampled[i][j].permute(1, 2, 0).numpy()
            ax.imshow((img - img.min()) / (img.max() - img.min()))
            ax.axis('off')
            ax.text(0.5, -0.1, str(j+1), transform=ax.transAxes, ha='center', va='top', fontsize=12, color='#2e3440')

        text_ax = plt.subplot(gs[i, samples])
        text_ax.axis('off')
        text_ax.text(0.5, 0.5, class_names[i], ha='center', va='center', fontsize=14, rotation=90, color='black', fontweight='bold')

        pred_ax = plt.subplot(gs[i, samples+1])
        pred_ax.axis('off')
        pred_ax.text(0.5, 0.95, "Model Predictions Across All Classes", ha='center', va='center', fontsize=12, fontweight='bold', color='#2e3440')

        for j in range(samples):
            p = preds[i][j]
            conf = probs[i][j][p].item()
            correct = (p == i)
            col = '#88C0D0' if correct else '#BF616A'
            h = 0.8 / samples
            y = 0.8 - j * h
            pred_ax.text(0.05, y - h/2, f"{j+1}:", ha='right', va='center', fontsize=10, color='#2e3440')
            pred_ax.add_patch(patches.Rectangle((0.1, y - h + 0.05), 0.8, h - 0.1, facecolor=col, alpha=0.3, edgecolor=col, linewidth=2, transform=pred_ax.transAxes))
            pred_ax.text(0.5, y - h/2, f"{class_names[p]}\nConf: {conf:.2f}", ha='center', va='center', fontsize=10, color='#2e3440')
            pred_ax.text(0.92, y - h/2, "✓" if correct else "✗", ha='center', va='center', fontsize=14, color=col, fontweight='bold')

    plt.tight_layout()
    plt.savefig('class_predictions_grid.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    train_path = "/kaggle/input/nature-922/inaturalist_12K/train"
    test_path = "/kaggle/input/nature-922/inaturalist_12K/val"

    best = {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'dense_size': 1024,
        'filter_size': [7, 5, 5, 3, 3],
        'activation': 'Mish',
        'filter_organisation': 'alt',
        'no_of_filters': 32,
        'data_augmentaion': 'No',
        'batch_normalization': 'Yes',
        'dropout': 0.2,
        'img_size': 224
    }

    t_dl, v_dl, te_dl, n_cls, cls_names = create_dataloaders(train_path, test_path, best['batch_size'], best['img_size'], best['data_augmentaion'])
    print("Training...")
    net = execute_training(best, t_dl, v_dl, n_cls)
    acc = get_accuracy(net, te_dl)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nGenerating visualization...")
    visualize(net, te_dl, cls_names)
