import numpy as np
from pathlib import Path
from PIL import Image

def flip_and_rot(arr):
    flipped = np.flip(arr, axis=0)
    lst = []
    for i in range(4):
        lst.append(np.rot90(arr, k=i+1))
        lst.append(np.rot90(flipped, k=i+1))
    return lst

def data_augmentation_from_folder(folder_path, save_folder_path):
    folder = Path(folder_path)

    arr_lst = []

    for file in folder.iterdir():
        arr = np.asarray(Image.open(file))
        lst = flip_and_rot(arr)
        for array in lst:
            arr_lst.append(array)
    
    for i in range(len(arr_lst)):
        img = Image.fromarray(arr_lst[i].astype(np.uint8))
        img.save(rf"{save_folder_path}\{i}.png")

if __name__ == "__main__":
    data_augmentation_from_folder(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\script and needles\improved_synthetic_2\train_raw_redone_before_aug", r"C:\Users\magfa\Documents\Master\Masteroppgave\data\script and needles\improved_synthetic_2\improved_synthetic_2_redone_15_04\train\raw")
    data_augmentation_from_folder(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\script and needles\improved_synthetic_2\train_label_redone_before_aug", r"C:\Users\magfa\Documents\Master\Masteroppgave\data\script and needles\improved_synthetic_2\improved_synthetic_2_redone_15_04\train\label")
    data_augmentation_from_folder(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\script and needles\improved_synthetic_2\test_raw_redone_before_aug", r"C:\Users\magfa\Documents\Master\Masteroppgave\data\script and needles\improved_synthetic_2\improved_synthetic_2_redone_15_04\test\raw")
    data_augmentation_from_folder(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\script and needles\improved_synthetic_2\test_label_redone_before_aug", r"C:\Users\magfa\Documents\Master\Masteroppgave\data\script and needles\improved_synthetic_2\improved_synthetic_2_redone_15_04\test\label")