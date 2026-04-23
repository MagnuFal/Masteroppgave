import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def IoU_and_mIoU(pred_argmax_array, label_array):
    label_array = label_array * 5
    overlay = label_array + pred_argmax_array
    unique = np.unique(overlay, return_counts=True)


if __name__ == "__main__":
    pred_array_path = r"C:\Users\magfa\Documents\Master\Masteroppgave\experiments\improved_dataset_2\improved_dataset_2_re_training_with_recorded_train_and_val_loss\re_test_test_set_predictions_argmax\0.png"
    label_path = r"C:\Users\magfa\Documents\Master\Masteroppgave\data\improved_synthetic_2_redone_15_04_test_set\label\32.png"

    img1 = Image.open(pred_array_path)
    img2 = Image.open(label_path)

    arr1 = np.asarray(img1)
    arr2 = np.asarray(img2)

    IoU_and_mIoU(arr1, arr2)