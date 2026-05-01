import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def IoU_and_mIoU(pred_argmax_array, label_array):
    label_array = label_array * 5
    overlay = label_array + pred_argmax_array
    unique = np.unique(overlay, return_counts=True)

def confusion_matrix_one_image(pred_path, label_path):
    pred_img = Image.open(pred_path)
    label_img = Image.open(label_path)

    pred = np.asarray(pred_img)
    label = np.asarray(label_img)

    pred_flatten = pred.flatten()
    label_flatten = label.flatten()

    disp = ConfusionMatrixDisplay.from_predictions(y_true=label_flatten, y_pred=pred_flatten, normalize = "true")
    plt.show()


if __name__ == "__main__":
    pred_array_path = r"C:\Users\magfa\Documents\Master\Masteroppgave\experiments\improved_dataset_2\improved_dataset_2_re_training_with_recorded_train_and_val_loss\re_test_test_set_predictions_argmax\0.png"
    label_path = r"C:\Users\magfa\Documents\Master\Masteroppgave\data\improved_synthetic_2_redone_15_04_test_set\label\32.png"

    confusion_matrix_one_image(pred_array_path, label_path)