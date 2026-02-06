from dataset_generation import SyntheticDataset
from .model import UNet
from torch.utils.data import DataLoader, random_split
from .train import optimization_loop, train_model, evaluate_model


if __name__ == "__main__":
    
    raw_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\synthetic_dataset\train\raw"
    label_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\synthetic_dataset\train\label"

    dataset = SyntheticDataset(raw_dir, label_dir)

    val_percent = 0.1

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, shuffle=True)
    val_loader = DataLoader(val_set, shuffle=False)


    model = UNet()
    sv_pt = r"C:\Users\magfa\Documents\Master\Masteroppgave\machine_learning\best_model_1.pth"
    optimization_loop(model, save_path=sv_pt, tr_loader=train_loader, vl_loader=val_loader)