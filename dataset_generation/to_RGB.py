import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
from pathlib import Path

def to_rgb(array):
    img = np.zeros((1790, 2560, 3))

    red_channel = (array == 0).astype(np.uint8) * 0
    green_channel = (array == 1).astype(np.uint8) * 255
    blue_channel = (array == 2).astype(np.uint8) * 255

    img[:, :, 0] = red_channel
    img[:, :, 1] = green_channel
    img[:, :, 2] = blue_channel

    return img

def folder_to_RGB(folder_path, save_folder_path):
    folder = Path(folder_path)
    save_folder = Path(save_folder_path)

    for file in folder.iterdir():
        arr = np.asarray(Image.open(file))
        rgb = to_rgb(arr).astype(np.uint8)
        img = Image.fromarray(rgb)
        img = img
        img.save(save_folder / file.name)

folder_to_RGB(r"C:\Users\magfa\Documents\Master\Masteroppgave\synthetic_dataset\test\label", r"C:\Users\magfa\Documents\Master\Masteroppgave\machine_learning\test_results\labels")