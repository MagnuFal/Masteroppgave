from dataset_class import SyntheticDataset, SyntheticDatasetAugmented
from model import UNet
from torch.utils.data import DataLoader, random_split
from train import optimization_loop, train_model, evaluate_model
from test import model_test
import torch
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    
    raw_dir = r"/cluster/home/magnufal/Master/Masteroppgave/data/dataset_3_plus_2/train/raw"
    label_dir = r"/cluster/home/magnufal/Master/Masteroppgave/data/dataset_3_plus_2/train/label"

    #raw_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\data\sanity_test\raw"
    #label_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\data\sanity_test\label"

    dataset = SyntheticDatasetAugmented(raw_dir, label_dir)

    val_percent = 0.1

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=1)
#
    training_weights = torch.tensor([0.589441, 4.623541, 3.650706])
#
    model = UNet()
    sv_pt = r"/cluster/home/magnufal/Master/Masteroppgave/machine_learning/dataset_3_plus_2_first_run.pth"
    #sv_pt = r"C:\Users\magfa\Documents\Master\Masteroppgave\machine_learning/sanity_test.pth"
    checkpoint = torch.load(r"/cluster/home/magnufal/Master/Masteroppgave/machine_learning/re_test_improved_dataset_2_with_train_val_loss_15_04_26.pth", weights_only=True, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimization_loop(model, save_path=sv_pt, tr_loader=train_loader, vl_loader=val_loader, weights=training_weights, epochs= 300)

    # --------------- Test ----------------------

    #raw_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\raw"
    #label_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label"
#
#
    #dataset = SyntheticDatasetAugmented(raw_dir, label_dir)
#
    #test_loader = DataLoader(dataset, shuffle=False)
#
    #model = UNet()
    #checkpoint = torch.load(r"C:\Users\magfa\Documents\Master\Masteroppgave\experiments\dataset_3\dataset_3_improved_7th_run_AdamW\dataset_3_improved_7th_run_AdamW.pth", weights_only=True, map_location=torch.device('cpu'))
    #model.load_state_dict(checkpoint['model_state_dict'])
#
    #model_test(model, test_loader, save_folder_path = r"C:\Users\magfa\Documents\Master\Masteroppgave\experiments\dataset_3\dataset_3_improved_7th_run_AdamW\predictions")
    #model = UNet()
    #summary(model, (1, 1, 224, 224))