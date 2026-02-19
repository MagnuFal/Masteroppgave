from dataset_generation import SyntheticDataset
from .model import UNet
from torch.utils.data import DataLoader, random_split
from .train import optimization_loop, train_model, evaluate_model
from .test import model_test
import torch
from torchinfo import summary


if __name__ == "__main__":
    
    raw_dir = r"/cluster/home/magnufal/Master/Masteroppgave/synthetic_dataset_2/train/raw"
    label_dir = r"/cluster/home/magnufal/Master/Masteroppgave/synthetic_dataset_2/train/label"

    dataset = SyntheticDataset(raw_dir, label_dir)

    val_percent = 0.1

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, shuffle=True)
    val_loader = DataLoader(val_set, shuffle=False)

    training_weights = torch.tensor([0.356657, 30.526744, 6.119014])

    model = UNet()
    sv_pt = r"/cluster/home/magnufal/Master/Masteroppgave/machine_learning/synthetic_dataset_2_from_scratch_with_weighted_loss.pth"
    optimization_loop(model, save_path=sv_pt, tr_loader=train_loader, vl_loader=val_loader, weights=training_weights)

    # --------------- Test ----------------------

    #raw_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\synthetic_dataset_2\test\raw"
    #label_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\synthetic_dataset_2\test\label"
#
    #dataset = SyntheticDataset(raw_dir, label_dir)
#
    #test_loader = DataLoader(dataset, shuffle=False)
#
    #model = UNet()
    #checkpoint = torch.load(r"C:\Users\magfa\Documents\Master\Masteroppgave\machine_learning\synthetic_dataset_2_with_random_weights.pth", weights_only=True, map_location=torch.device('cpu'))
    #model.load_state_dict(checkpoint['model_state_dict'])
#
    #model_test(model, test_loader, save_folder_path = r"C:\Users\magfa\Documents\Master\Masteroppgave\machine_learning\predictions\synthetic_dataset_2_with_random_weights")

    #model = UNet()
    #summary(model, (1, 1, 224, 224))