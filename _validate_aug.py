# -*- coding: utf-8 -*-
"""Fast CPU validation of the targeted-augmentation pipeline on a small real subset."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
OVERSAMPLE = {6: 4, 4: 3, 0: 2, 2: 2}
HARD_CLASSES = set(OVERSAMPLE.keys())

raw_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=None)
raw_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=None)
all_images = torch.cat([raw_train.data, raw_test.data], dim=0)
all_labels = torch.cat([raw_train.targets, raw_test.targets], dim=0)
print("all_images", all_images.shape, all_images.dtype, "all_labels", all_labels.shape)

all_indices = np.arange(len(all_labels))
labels_np = all_labels.numpy()
temp_indices, test_indices, y_temp, y_test = train_test_split(
    all_indices, labels_np, test_size=0.2, stratify=labels_np, random_state=42)
train_indices, val_indices, _, _ = train_test_split(
    temp_indices, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
print("split", len(train_indices), len(val_indices), len(test_indices))

normal_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
strong_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.12)),
])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

class TargetedAugDataset(Dataset):
    def __init__(self, images, labels, indices, normal_tf, hard_tf, hard_classes):
        self.images = images
        self.labels = labels
        self.indices = list(indices)
        self.normal_tf = normal_tf
        self.hard_tf = hard_tf
        self.hard_classes = set(hard_classes)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        label = int(self.labels[idx])
        img = Image.fromarray(self.images[idx].numpy(), mode='L')
        tf = self.hard_tf if label in self.hard_classes else self.normal_tf
        return tf(img), label

# oversample on a SMALL subset for speed
small_train = train_indices[:400]
oversampled = []
for idx in small_train:
    factor = OVERSAMPLE.get(int(all_labels[idx]), 1)
    oversampled.extend([idx] * factor)
print("small_train", len(small_train), "-> oversampled", len(oversampled))

train_ds = TargetedAugDataset(all_images, all_labels, oversampled, normal_transform, strong_transform, HARD_CLASSES)
val_ds = TargetedAugDataset(all_images, all_labels, test_indices[:200], eval_transform, eval_transform, set())
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

xb, yb = next(iter(train_loader))
print("batch", xb.shape, xb.dtype, "min/max", float(xb.min()), float(xb.max()), "labels", yb.shape)

class CNN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 10))

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))

device = torch.device("cpu")
model = CNN2D().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# one train epoch
model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
print("train epoch ok, last loss", float(loss))

# best-state checkpoint pattern
best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
print("checkpoint clone/load ok")

# predictions + confusion
model.eval()
preds, trues = [], []
with torch.no_grad():
    for images, labels in val_loader:
        out = model(images.to(device))
        preds.append(out.max(1)[1].cpu())
        trues.append(labels)
y_pred = torch.cat(preds); y_true = torch.cat(trues)
cm = confusion_matrix(y_true.numpy(), y_pred.numpy(), labels=list(range(10)))
print("confusion ok", cm.shape)
_ = classification_report(y_true.numpy(), y_pred.numpy(), labels=list(range(10)),
                          target_names=class_names, digits=4, zero_division=0)
print("classification_report ok")
print("VALIDATION_OK")
