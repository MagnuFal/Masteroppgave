from dataset_generation import SyntheticDataset
from .model import UNet
from torch.utils.data import DataLoader, random_split
from .train import optimization_loop, train_model, evaluate_model
from .test import model_test
import torch
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    
    raw_dir = r"/cluster/home/magnufal/Master/Masteroppgave/dataset_3/train/raw"
    label_dir = r"/cluster/home/magnufal/Master/Masteroppgave/dataset_3/train/label"

    dataset = SyntheticDataset(raw_dir, label_dir)

    val_percent = 0.1

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, shuffle=True)
    val_loader = DataLoader(val_set, shuffle=False)
#
    training_weights = torch.tensor([0.345980, 25.150191, 14.306591])
#
    model = UNet()
    sv_pt = r"/cluster/home/magnufal/Master/Masteroppgave/machine_learning/dataset_3_second_run.pth"
    checkpoint = torch.load(r"/cluster/home/magnufal/Master/Masteroppgave/weights/dataset_3_first_run.pth", weights_only=True, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimization_loop(model, save_path=sv_pt, tr_loader=train_loader, vl_loader=val_loader, weights=training_weights)

    # --------------- Test ----------------------

    #raw_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\improved_dataset_2\test\raw"
    #label_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\improved_dataset_2\test\label"

    #raw_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\test_inference\raw"
    #label_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\test_inference\label"
#
    #dataset = SyntheticDataset(raw_dir, label_dir)
#
    #test_loader = DataLoader(dataset, shuffle=False)
#
    #model = UNet()
    #checkpoint = torch.load(r"C:\Users\magfa\Documents\Master\Masteroppgave\machine_learning\weights\improved_dataset_2_first_run.pth", weights_only=True, map_location=torch.device('cpu'))
    #model.load_state_dict(checkpoint['model_state_dict'])
#
    #model_test(model, test_loader, save_folder_path = r"C:\Users\magfa\Documents\Master\Masteroppgave\machine_learning\predictions\test_inference")

    #model = UNet()
    #summary(model, (1, 1, 224, 224))