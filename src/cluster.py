from dataset_class import SyntheticDataset, SyntheticDatasetAugmented
from model import UNet, DeepUNet
from torch.utils.data import DataLoader, random_split
from train import optimization_loop, train_model, evaluate_model
from test import model_test
import torch
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    
    #raw_dir = r"/cluster/home/magnufal/Master/Masteroppgave/data/dataset_3_improved/train/raw"
    #label_dir = r"/cluster/home/magnufal/Master/Masteroppgave/data/dataset_3_improved/train/label"
#
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
    #training_weights = torch.tensor([0.343073, 30.572882, 19.060775])
##
    #model = UNet()
    #sv_pt = r"/cluster/home/magnufal/Master/Masteroppgave/machine_learning/from_first_run_checkpoint_decreased_lr_12_05.pth"
    ##sv_pt = r"C:\Users\magfa\Documents\Master\Masteroppgave\machine_learning/sanity_test.pth"
    #checkpoint = torch.load(r"/cluster/home/magnufal/Master/Masteroppgave/machine_learning/dataset_3_improved_first_run.pth", weights_only=True, map_location=torch.device(device))
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimization_loop(model, save_path=sv_pt, tr_loader=train_loader, vl_loader=val_loader, weights=None, epochs= 200, lr=10**-4)

    # --------------- Test ----------------------

    #raw_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\raw"
    #label_dir = r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label"
    raw_dir = r"/cluster/home/magnufal/Master/Masteroppgave/data/two step model phase differentiation/train/raw"
    label_dir = r"/cluster/home/magnufal/Master/Masteroppgave/data/two step model phase differentiation/train/label"

    dataset = SyntheticDatasetAugmented(raw_dir, label_dir)

    test_loader = DataLoader(dataset, shuffle=False)

    model = UNet(n_classes=2)
    checkpoint = torch.load(r"C:\Users\magfa\Documents\Master\Masteroppgave\experiments\two-step model\phase extraction run 1\two_step_model_phase_extraction_run_1.pth", weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    model_test(model, test_loader, save_folder_path = r"/cluster/home/magnufal/Master/Masteroppgave/experiments/predictions on training set of phase extraction")
    #model = UNet()
    #summary(model, (1, 1, 224, 224))