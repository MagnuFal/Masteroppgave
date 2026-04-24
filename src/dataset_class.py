import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from pathlib import Path
import os
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import albumentations as A
import numpy as np
from PIL import Image
import cv2

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
        raw_iter_folder = list(Path(self.raw_dir).iterdir())
        raw_img_path = raw_iter_folder[index]
        raw_image = read_image(raw_img_path)
        raw_image = raw_image.float() / 255.0 # This line should maybe be removed since images are already [0, 1]. Could slow down training if kept.
        raw_image = raw_image.mean(dim=0, keepdim=True) # Greyscale. Is this line needed now?
        label_iter_folder = list(Path(self.label_dir).iterdir())
        label_image_path = label_iter_folder[index]
        label_image = read_image(label_image_path)
        label_image = label_image.squeeze(0) # This probably still needs to be here
        label_image = label_image.long() # This one is maybe superficial
        if self.transform:
            raw_image = self.transform(raw_image)
        if self.label_transform:
            label_image = self.label_transform(label_image)
        #print("raw_image", raw_image.shape, raw_image.dtype)
        #print("label_image", label_image.shape, label_image.dtype, label_image.min(), label_image.max())

        return raw_image, label_image
    

transform_pipeline = A.Compose([
    A.OneOf([
        A.RandomCrop(width=1500, height=1500, p=0.25),
        A.RandomCrop(width=1000, height=1000, p=0.25),
        A.RandomCrop(width=700, height=700, p=0.25),
        A.RandomCrop(width=400, height=400, p=0.25),], p=0.7
    ),
    A.RandomRotate90(p=1),
    A.VerticalFlip(p=0.5),
])
    
class SyntheticDatasetAugmented(Dataset):
    def __init__(self, raw_dir, label_dir):
        self.raw_dir = raw_dir
        self.label_dir = label_dir

    def __len__(self):
        folder = Path(self.raw_dir)
        return len([p for p in folder.iterdir() if p.is_file()])
    
    def __getitem__(self, index): # This function might need cleaning up, after implemented changes in opencv.py 
        raw_iter_folder = list(Path(self.raw_dir).iterdir())
        raw_img_path = raw_iter_folder[index]
        raw_img = Image.open(raw_img_path)
        raw_image = np.asarray(raw_img)

        #zero_array = np.zeros((385, 2560))

        #raw_image = np.concatenate([raw_image, zero_array], axis=0)
        #raw_image = np.concatenate([zero_array, raw_image], axis=0)

        label_iter_folder = list(Path(self.label_dir).iterdir())
        label_image_path = label_iter_folder[index]
        label_img = Image.open(label_image_path)
        label_image = np.asarray(label_img)

        #label_image = np.concatenate([label_image, zero_array], axis=0)
        #label_image = np.concatenate([zero_array, label_image], axis=0)

        raw_image = raw_image[:, :, np.newaxis]
        label_image = label_image[:, :, np.newaxis]

        transformed_data = transform_pipeline(image = raw_image, mask = label_image)
        raw_image = transformed_data["image"]
        label_image = transformed_data["mask"]

        raw_image = torch.tensor(np.transpose(raw_image, (2, 0, 1)), dtype=torch.float32)
        label_image = torch.tensor(np.transpose(label_image, (2, 0, 1)), dtype=torch.long)

        label_image = torch.squeeze(label_image, dim = 0)

        print(raw_image.shape)
        print(label_image.shape)

        #label_image = label_image.squeeze(0) # This probably still needs to be here
        #label_image = label_image.long() # This one is maybe superficial
#
        #raw_image = raw_image.float / 255.0

        return raw_image, label_image