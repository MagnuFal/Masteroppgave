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

dataset = SyntheticDataset(raw_dir, label_dir)

val_percent = 0.1

n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val

train_set, val_set = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_set, shuffle=True)
val_loader = DataLoader(val_set, shuffle=False)

def train_model(model, lr = 10**-3, batch_size = 8, loss_fn = nn.CrossEntropyLoss()):

    optimizer = optim.RMSprop(model.parameters(), lr)

    size = len(train_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        pred = model(X)
        print(pred.shape, pred.dtype)
        print(y.shape, y.dtype, y.min().item(), y.max().item())
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return optimizer.state_dict()

def evaluate_model(model, loss_fn = nn.CrossEntropyLoss()):
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

    return test_loss  

def optimization_loop(model, save_path, epochs = 20):
    best_val_loss = 100
    for k in range(epochs):
        print(f"---------- Epoch {k + 1} ----------")
        opt_state_dict = train_model(model)
        val_loss = evaluate_model(model, bvl=best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch" : k + 1,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : opt_state_dict,
                "loss" : val_loss
            }, save_path)
            print(f"Best Validation Loss Updated: {val_loss}")
    print("Finished!")

if __name__ == "__main__":
    model = UNet()
    sv_pt = r"C:\Users\magfa\Documents\Master\Masteroppgave\machine_learning\best_model_1.pth"
    optimization_loop(model, save_path=sv_pt)