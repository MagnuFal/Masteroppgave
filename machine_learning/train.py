from dataset_generation import dataset_class
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from pathlib import Path
import os
from torchvision.io import decode_image
from torch.utils.data import DataLoader, random_split

raw_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\synthetic_dataset\raw"
label_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\synthetic_dataset\label"

def train_model(model, device, epochs = 10, lr = 10**-3, batch_size = 8, val_percent = 0.1):
    
    dataset = dataset_class.SyntheticDataset(raw_dir, label_dir)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, shuffle=True)
    train_loader = DataLoader(val_set, shuffle=False)

    