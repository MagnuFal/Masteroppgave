from dataset_class import SyntheticDataset
from model import UNet
import torch
from torch import optim
import torch.nn as nn
from pathlib import Path
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, tr_loader, optimizer, scheduler, batch_size = 1, loss_fn = nn.CrossEntropyLoss()):

    size = len(tr_loader.dataset)
    model.train()

    for batch, (X, y) in enumerate(tr_loader):
        X = torch.tensor(X, requires_grad=True).to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        if batch % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    scheduler.step()
    return optimizer.state_dict(), loss, X.grad

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

def save_best_model(model, epoch, opt_state, best_loss, save_path):
    torch.save({
            "epoch" : epoch + 1,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : opt_state,
            "loss" : best_loss
            }, save_path)

def optimization_loop(model, save_path,  tr_loader, vl_loader, epochs = 300, weights = None, lr = 10**-3):
    model = model.to(device)
    weights = weights.to(device)
    file_path = Path(save_path)
    loss_log = f"{file_path.stem}.txt"
    with open(loss_log, "a") as f:
        f.write(f"Epoch, Val_Loss, Train_Loss, lr, X.grad\n")
    best_val_loss = 0

    optimizer = optim.AdamW(model.parameters(), lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    for k in range(epochs):
        print(f"---------- Epoch {k + 1} ----------")
        opt_state_dict, train_loss, input_grad = train_model(model, tr_loader, optimizer=optimizer, scheduler=scheduler, loss_fn=nn.CrossEntropyLoss(weight=weights))
        val_loss = evaluate_model(model, vl_loader) # Removed bvl=best_val_loss
        with open(loss_log, "a") as f:
            f.write(f"{k + 1}, {val_loss}, {train_loss}, {opt_state_dict['param_groups'][0]['lr']}, {input_grad}\n")
        if k == 0:
            best_val_loss = val_loss
            save_best_model(model, k, opt_state_dict, best_val_loss, save_path)
            print(f"Best Validation Loss Updated: {best_val_loss}")
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_model(model, k, opt_state_dict, best_val_loss, save_path)
            print(f"Best Validation Loss Updated: {best_val_loss}")
    print(f"Finished! - Best Validation Loss: {best_val_loss}")