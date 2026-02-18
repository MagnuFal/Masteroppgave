from dataset_generation import SyntheticDataset
from .model import UNet
import torch
from torch import optim
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, tr_loader, lr = 10**-3, batch_size = 8, loss_fn = nn.CrossEntropyLoss()):

    optimizer = optim.RMSprop(model.parameters(), lr)

    size = len(tr_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(tr_loader):
        X = X.to(device)
        y = y.to(device)
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

def evaluate_model(model, vl_loader, loss_fn = nn.CrossEntropyLoss()):
    model.eval()
    size = len(vl_loader.dataset)
    num_batches = len(vl_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in vl_loader: # Pass paa at det ikke blir feil siden
            X = X.to(device) # jeg allerede har brukt X, y
            y = y.to(device) 
            pred = model(X)     
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss  

def optimization_loop(model, save_path,  tr_loader, vl_loader, epochs = 30, weights = None):
    model = model.to(device)
    best_val_loss = 100
    for k in range(epochs):
        print(f"---------- Epoch {k + 1} ----------")
        opt_state_dict = train_model(model, tr_loader, loss_fn=nn.CrossEntropyLoss(weights))
        val_loss = evaluate_model(model, vl_loader) # Removed bvl=best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch" : k + 1,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : opt_state_dict,
                "loss" : val_loss
            }, save_path)
            print(f"Best Validation Loss Updated: {val_loss}")
    print(f"Finished! - Best Validation Loss: {val_loss}")