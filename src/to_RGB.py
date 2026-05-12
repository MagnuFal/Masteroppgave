import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
from pathlib import Path

def to_rgb(array):
    h, w = array.shape

    img = np.zeros((h, w, 3), dtype=np.uint8)

    red_channel = (array == 0).astype(np.uint8) * 0
    green_channel = (array == 1).astype(np.uint8) * 255
    blue_channel = (array == 2).astype(np.uint8) * 255

    img[:, :, 0] = red_channel
    img[:, :, 1] = green_channel
    img[:, :, 2] = blue_channel

    return img

def to_bw(array):

    array = (array * 255).astype(np.uint8)

    return array

def folder_to_RGB(folder_path, save_folder_path):
    folder = Path(folder_path)
    save_folder = Path(save_folder_path)

    for file in folder.iterdir():
        arr = np.asarray(Image.open(file))
        rgb = to_rgb(arr).astype(np.uint8)
        img = Image.fromarray(rgb)
        img = img
        img.save(save_folder / file.name)

def folder_to_bw(folder_path, save_folder_path):
    folder = Path(folder_path)
    save_folder = Path(save_folder_path)
    

    for file in folder.iterdir():
        arr = np.asarray(Image.open(file))
        bw = to_bw(arr).astype(np.uint8)
        img = Image.fromarray(bw)
        img.save(save_folder / file.name)

if __name__ == "__main__":
    folder_to_bw(r"C:\Users\magfa\Documents\Master\Masteroppgave\data\synthetic_dataset_1\train\raw", r"C:\Users\magfa\Documents\Master\Masteroppgave\data\synthetic_dataset_1\train\raw_v")