import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from pathlib import Path
import os
from torchvision.io import read_image
from torch.utils.data import DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, raw_dir, label_dir, transfrom = None, label_transform = None):
        self.raw_dir = raw_dir
        self.label_dir = label_dir
        self.transform = transfrom
        self.label_transform = label_transform

    def __len__(self):
        folder = Path(self.raw_dir)
        return len([p for p in folder.iterdir() if p.is_file()])
    
    def __getitem__(self, index):
        raw_img_path = os.path.join(self.raw_dir, f"{index}.png")
        raw_image = read_image(raw_img_path)
        raw_image = raw_image.float() / 255.0
        label_image_path = os.path.join(self.label_dir, f"{index}.png")
        label_image = read_image(label_image_path)
        if self.transform:
            raw_image = self.transform(raw_image)
        if self.label_transform:
            label_image = self.label_transform(label_image)
        return raw_image, label_image