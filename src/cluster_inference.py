from dataset_class import SyntheticDataset, SyntheticDatasetAugmented
from model import UNet, DeepUNet
from torch.utils.data import DataLoader, random_split
from train import optimization_loop, train_model, evaluate_model
from test import model_test
import torch
from torchinfo import summary


    # --------------- Test ----------------------
if __name__ == "__main__":
    raw_dir = r"/cluster/home/magnufal/Master/Masteroppgave/synthetic_dataset/test/raw"
    label_dir = r"/cluster/home/magnufal/Master/Masteroppgave/synthetic_dataset/test/label"
    #raw_dir1 = r"/cluster/home/magnufal/Master/Masteroppgave/data/dataset_3_improved_resized/test/raw"
    #label_dir1 = r"/cluster/home/magnufal/Master/Masteroppgave/data/dataset_3_improved_resized/test/label"
#
    #raw_dir2 = r"/cluster/home/magnufal/Master/Masteroppgave/data/improved_synthetic_2_redone_15_04/raw"
    #label_dir2 = r"/cluster/home/magnufal/Master/Masteroppgave/data/improved_synthetic_2_redone_15_04/label"
#
    #raw_dir3 = r"/cluster/home/magnufal/Master/Masteroppgave/data/dataset_3_improved/test/raw"
    #label_dir3 = r"/cluster/home/magnufal/Master/Masteroppgave/data/dataset_3_improved/test/label"
    #raw_dir2 = r"/cluster/home/magnufal/Master/Masteroppgave/experiments/phase extraction and differentiation run 1 end-to-end test/extraction prediction"

    dataset = SyntheticDatasetAugmented(raw_dir, label_dir)

    test_loader = DataLoader(dataset, shuffle=False)

    model = UNet()
    checkpoint = torch.load(r"/cluster/home/magnufal/Master/Masteroppgave/experiments/dataset_1_re_training_with_val_train_loss/dataset_1_re_training_with_val_train_loss_run_1.pth", weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    model_test(model, test_loader, save_folder_path = r"/cluster/home/magnufal/Master/Masteroppgave/experiments/dataset_1_re_training_with_val_train_loss/predictions_argmax")
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