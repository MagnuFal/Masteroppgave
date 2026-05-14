import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import segmentation_models_pytorch as smp
import torchvision.io as io
from rgb_prediction_to_8_bit import to_rgb
from pathlib import Path



def confusion_matrix_one_image(pred_path, label_path):
    pred_img = Image.open(pred_path)
    label_img = Image.open(label_path)

    pred = np.asarray(pred_img)
    label = np.asarray(label_img)

    pred_flatten = pred.flatten()
    label_flatten = label.flatten()

    disp = ConfusionMatrixDisplay.from_predictions(y_true=label_flatten, y_pred=pred_flatten, normalize = "true", cmap="Blues")
    plt.show()

def confusion_matrix_from_folder(pred_folder_path, label_folder_path):
    pred_folder = Path(pred_folder_path)
    label_folder = Path(label_folder_path)

    pred_lst = [np.asarray(Image.open(file)) for file in pred_folder.iterdir()]
    label_lst = [np.asarray(Image.open(file)) for file in label_folder.iterdir()]

    total_pred_arr = np.zeros_like(pred_lst[0])
    total_label_arr = np.zeros_like(label_lst[0])

    for arr in pred_lst:
        total_pred_arr = np.concatenate((total_pred_arr.flatten(), arr.flatten()))

    for arr in label_lst:
        total_label_arr = np.concatenate((total_label_arr.flatten(), arr.flatten()))

    class_names = ["No Phase", "Platelet", "Script"]

    ConfusionMatrixDisplay.from_predictions(y_true=total_label_arr, y_pred=total_pred_arr, normalize = "true", cmap="Blues", display_labels=class_names)

    plt.savefig(r"C:\Users\magfa\Documents\Master\Masteroppgave\figures\MasterFigures\cm_improved_dataset_2.png", dpi = 1300, bbox_inches="tight")   



if __name__ == "__main__":
    #img1 = Image.open(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label\Mask of org_6nbr_7_upscaled_6pm.png")
    #img2 = Image.open(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label\Mask of org_7nbr_1_upscaled_6pm.png")
    #img3 = Image.open(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label\Mask of org_7nbr_2_upscaled_6m.png")
    #img4 = Image.open(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label\Mask of org_7nbr_6_upscaled_2ps.png")
    #img5 = Image.open(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label\Mask of org_7nbr_6_upscaled_6m.png")
    #img6 = Image.open(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label\Mask of org_7nbr_11_upscaled_6pm.png")
    #img7 = Image.open(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label\Mask of org_7nbr_12_upscaled_2ps.png")
    #img8 = Image.open(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label\Mask of org_8nbr_2_upscaled_6pm.png")
    #img9 = Image.open(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label\Mask of org_10nbr_5_upscaled_6ps.png")
    #img10 = Image.open(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label\Mask of org_10nbr_12_upscaled_2m.png")
    #img11 = Image.open(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label\Mask of org_12nbr_7_upscaled_2m.png")
    #img12 = Image.open(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\dataset_3_improved\test\label\Mask of org_7nbr_6_upscaled_6pm.png")
#
    #
    #arr1 = np.asarray(img1)
    #arr2 = np.asarray(img2)
    #arr3 = np.asarray(img3)
    #arr4 = np.asarray(img4)
    #arr5 = np.asarray(img5)
    #arr6 = np.asarray(img6)
    #arr7 = np.asarray(img7)
    #arr8 = np.asarray(img8)
    #arr9 = np.asarray(img9)
    #arr10 = np.asarray(img10)
    #arr11 = np.asarray(img11)
    #arr12 = np.asarray(img12)
#
    #fig, ax = plt.subplots(2, 6)
#
    #flatten = ax.flatten()
#
    #flatten[0].imshow(arr1)
    #flatten[1].imshow(arr2)
    #flatten[2].imshow(arr3)
    #flatten[3].imshow(arr4)
    #flatten[4].imshow(arr5)
    #flatten[5].imshow(arr6)
    #flatten[6].imshow(arr7)
    #flatten[7].imshow(arr8)
    #flatten[8].imshow(arr9)
    #flatten[9].imshow(arr10)
    #flatten[10].imshow(arr11)
    #flatten[11].imshow(arr12)
#
    #plt.show()



    confusion_matrix_from_folder(r"C:\Users\magfa\Documents\Master\Masteroppgave\experiments\improved_dataset_2\improved_dataset_2_re_training_with_recorded_train_and_val_loss\re_test_test_set_predictions_argmax",
                                 r"C:\Users\magfa\Documents\Master\Masteroppgave\data\improved_synthetic_2_redone_15_04_test_set\label")