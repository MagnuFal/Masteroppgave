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
    
    def __getitem__(self, index): # This function might need cleaning up, after implemented changes in opencv.py 
        raw_img_path = os.path.join(self.raw_dir, f"{index}.png")
        raw_image = read_image(raw_img_path)
        raw_image = raw_image.float() / 255.0 # This line should maybe be removed since images are already [0, 1]. Could slow down training if kept.
        raw_image = raw_image.mean(dim=0, keepdim=True) # Greyscale. Is this line needed now?
        label_image_path = os.path.join(self.label_dir, f"{index}.png")
        label_image = read_image(label_image_path)
        label_image = label_image.squeeze(0) # This probably still needs to be here
        label_image = label_image.long() # This one is maybe superficial
        if self.transform:
            raw_image = self.transform(raw_image)
        if self.label_transform:
            label_image = self.label_transform(label_image)
        print("raw_image", raw_image.shape, raw_image.dtype)
        print("label_image", label_image.shape, label_image.dtype, label_image.min(), label_image.max())

        return raw_image, label_image