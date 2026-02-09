from dataset_generation import SyntheticDataset
from .model import UNet
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_test(model, tst_loader, loss_fn = nn.CrossEntropyLoss()):
    model.eval()
    size = len(tst_loader.dataset)
    num_batches = len(tst_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for index, (X, y) in enumerate(tst_loader): 
            X = X.to(device) 
            y = y.to(device) 
            pred = model(X) # Vil lagre prediksjonen som et 8-bit bilde
            pred_array = np.asarray(pred)
            arr = pred_array[0]
            rgb = np.transpose(arr)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            rgb = (rgb * 255).astype(np.uint8)
            #pred_array = pred_array.mean(dim=0, keepdim=True)
            im = Image.fromarray(rgb)
            im.save(rf"C:\Users\magfa\Documents\Master\Masteroppgave\machine_learning\predictions\{index}.png")