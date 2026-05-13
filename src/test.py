from dataset_class import SyntheticDataset
from model import UNet
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_test(model, tst_loader, save_folder_path, loss_fn = nn.CrossEntropyLoss()):
    model.eval()
    model.to(device)
    size = len(tst_loader.dataset)
    num_batches = len(tst_loader)
    test_loss, correct = 0, 0
    folder_path = Path(save_folder_path)

    with torch.no_grad():
        for index, (X, y) in enumerate(tst_loader): 
            X = X.to(device) 
            y = y.to(device) 
            pred = model(X) # Vil lagre prediksjonen som et 8-bit bilde
            pred_array = pred.detach().cpu().numpy()
            arr = pred_array[0]
            rgb = np.transpose(arr)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            rgb = (rgb * 255).astype(np.uint8)
            #pred_array = pred_array.mean(dim=0, keepdim=True)
            im = Image.fromarray(rgb)
            im.save(folder_path / f"{index}.png")