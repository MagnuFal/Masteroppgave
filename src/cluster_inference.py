from dataset_class import SyntheticDataset, SyntheticDatasetAugmented
from model import UNet, DeepUNet
from torch.utils.data import DataLoader, random_split
from train import optimization_loop, train_model, evaluate_model
from test import model_test
import torch
from torchinfo import summary


    # --------------- Test ----------------------
if __name__ == "__main__":
    raw_dir = r"/cluster/home/magnufal/Master/Masteroppgave/data/improved_synthetic_2_redone_15_04_resized/test/raw"
    label_dir = r"/cluster/home/magnufal/Master/Masteroppgave/data/improved_synthetic_2_redone_15_04_resized/test/label"

    dataset = SyntheticDatasetAugmented(raw_dir, label_dir)

    test_loader = DataLoader(dataset, shuffle=False)

    model = UNet()
    checkpoint = torch.load(r"/cluster/home/magnufal/Master/Masteroppgave/experiments/improved_synthetic_2_redone_15_04_resized/run1.pth", weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    model_test(model, test_loader, save_folder_path = r"/cluster/home/magnufal/Master/Masteroppgave/experiments/improved_synthetic_2_redone_15_04_resized/predictions_argmax")
#
    #dataset = SyntheticDatasetAugmented(raw_dir3, label_dir3)
#
    #test_loader = DataLoader(dataset, shuffle=False)
#
    #model = UNet()
    #checkpoint = torch.load(r"/cluster/home/magnufal/Master/Masteroppgave/machine_learning/dataset_3_improved_first_run.pth", weights_only=True, map_location=torch.device('cpu'))
    #model.load_state_dict(checkpoint['model_state_dict'])
#
    #model_test(model, test_loader, save_folder_path = r"/cluster/home/magnufal/Master/Masteroppgave/experiments/dataset_3_improved_14_05_26/predictions_argmax_wout_logits_norm")

    #model = UNet()
    #summary(model, (1, 1, 224, 224))