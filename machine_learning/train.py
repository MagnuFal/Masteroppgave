from dataset_generation import SyntheticDataset
from .model import UNet
import torch
from torch import optim
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from pathlib import Path
import os
from torchvision.io import decode_image
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

raw_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\synthetic_dataset\train\raw"
label_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\synthetic_dataset\train\label"

def train_model(model, epochs = 100, lr = 10**-3, batch_size = 8, val_percent = 0.1):
    
    dataset = SyntheticDataset(raw_dir, label_dir)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, shuffle=True)
    val_loader = DataLoader(val_set, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.rmsprop(model.parameters(), lr)

    size = len(train_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    model.eval()
    size = len(val_loader.dataset)
    num_batches = len(val_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in val_loader: # Pass paa at det ikke blir feil siden
            pred = model(X)     # jeg allerede har brukt X, y
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    model = UNet()

    train_model(model)