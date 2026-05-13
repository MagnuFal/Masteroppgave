from dataset_class import SyntheticDataset, SyntheticDatasetAugmented
from model import UNet, DeepUNet
from torch.utils.data import DataLoader, random_split
from train import optimization_loop, train_model, evaluate_model
from test import model_test
import torch
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    
    ##raw_dir = r"/cluster/home/magnufal/Master/Masteroppgave/data/dataset_3_improved/train/raw"
    ##label_dir = r"/cluster/home/magnufal/Master/Masteroppgave/data/dataset_3_improved/train/label"
    #raw_dir = r"/cluster/home/magnufal/Master/Masteroppgave/data/two step model phase differentiation/train/raw"
    #label_dir = r"/cluster/home/magnufal/Master/Masteroppgave/data/two step model phase differentiation/train/label"
    ##raw_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\data\sanity_test\raw"
    ##label_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\data\sanity_test\label"
#
    #dataset = SyntheticDatasetAugmented(raw_dir, label_dir)
#
    #val_percent = 0.1
#
    #n_val = int(len(dataset) * val_percent)
    #n_train = len(dataset) - n_val
#
    #train_set, val_set = random_split(dataset, [n_train, n_val])
#
    #train_loader = DataLoader(train_set, shuffle=True, batch_size=1)
    #val_loader = DataLoader(val_set, shuffle=False, batch_size=1)
##
    ##training_weights = torch.tensor([0.343073, 30.572882, 19.060775])
##
    #model = UNet()
    #sv_pt = r"/cluster/home/magnufal/Master/Masteroppgave/machine_learning/phase_differentiation_first_run_13_05.pth"
    ##sv_pt = r"C:\Users\magfa\Documents\Master\Masteroppgave\machine_learning/sanity_test.pth"
    #checkpoint = torch.load(r"/cluster/home/magnufal/Master/Masteroppgave/machine_learning/re_test_improved_dataset_2_with_train_val_loss_15_04_26.pth", weights_only=True, map_location=torch.device(device))
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimization_loop(model, save_path=sv_pt, tr_loader=train_loader, vl_loader=val_loader, weights=None, epochs= 200, lr=10**-4)

    # --------------- Test ----------------------

    #raw_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\data\two-step model datasets\step 2 phase differentiation\test\raw"
    #label_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\data\two-step model datasets\step 2 phase differentiation\test\label"
    raw_dir1 = r"/cluster/home/magnufal/Master/Masteroppgave/experiments/phase extraction and differentiation run 1 end-to-end test/raw"
    label_dir1 = r"/cluster/home/magnufal/Master/Masteroppgave/experiments/phase extraction and differentiation run 1 end-to-end test/raw"

    dataset = SyntheticDatasetAugmented(raw_dir1, label_dir1)

    test_loader = DataLoader(dataset, shuffle=False)

    model = UNet()
    checkpoint = torch.load(r"/cluster/home/magnufal/Master/Masteroppgave/machine_learning/two_step_model_phase_extraction_run_1.pth", weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    model_test(model, test_loader, save_folder_path = r"/cluster/home/magnufal/Master/Masteroppgave/experiments/phase extraction and differentiation run 1 end-to-end test/extraction prediction")

    raw_dir2 = r"/cluster/home/magnufal/Master/Masteroppgave/experiments/phase extraction and differentiation run 1 end-to-end test/extraction prediction"

    dataset = SyntheticDatasetAugmented(raw_dir2, label_dir1)

    test_loader = DataLoader(dataset, shuffle=False)

    model = UNet()
    checkpoint = torch.load(r"/cluster/home/magnufal/Master/Masteroppgave/machine_learning/re_test_improved_dataset_2_with_train_val_loss_15_04_26.pth", weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    model_test(model, test_loader, save_folder_path = r"/cluster/home/magnufal/Master/Masteroppgave/experiments/phase extraction and differentiation run 1 end-to-end test/differentiation prediction")

    #model = UNet()
    #summary(model, (1, 1, 224, 224))