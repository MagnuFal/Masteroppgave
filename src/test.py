from dataset_class import SyntheticDataset
from model import UNet
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_test(model, tst_loader, save_folder_path):
    model.eval()
    model.to(device)
    folder_path = Path(save_folder_path)

    with torch.no_grad():
        for index, (X, y, name) in enumerate(tst_loader): 
            X = X.to(device) 
            y = y.to(device) 
            pred = model(X)
            pred_array = pred.detach().cpu().numpy()
            arr = pred_array[0]
            arr = np.argmax(arr, axis=0).astype(np.uint8)
            im = Image.fromarray(arr)
            im.save(folder_path / name[0])