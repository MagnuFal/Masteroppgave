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