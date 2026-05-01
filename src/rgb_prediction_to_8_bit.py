import numpy as np
from pathlib import Path
from PIL import Image
from to_RGB import to_rgb

def rgb_to_8_bit(three_channel_tensor, axis = 2):
    pred_array = np.asarray(three_channel_tensor)
    
    return np.argmax(pred_array, axis=axis)

def predictions_argmax_from_folder_rgb(folder_path, save_folder_path1, save_folder_path2):
    folder = Path(folder_path)
    for file in folder.iterdir():
        img = Image.open(file)
        arr = np.asarray(img)
        eight_bit = rgb_to_8_bit(arr).astype(np.uint8)
        argmax = Image.fromarray(eight_bit)
        rgb = to_rgb(eight_bit).astype(np.uint8)
        argmax_v = Image.fromarray(rgb)
        argmax.save(f"{save_folder_path1}\{file.stem}.png")
        argmax_v.save(f"{save_folder_path2}\{file.stem}.png")

def predictions_argmax_from_folder_binary(folder_path, save_folder_path1, save_folder_path2):
    folder = Path(folder_path)
    for file in folder.iterdir():
        img = Image.open(file)
        arr = np.asarray(img)
        eight_bit = rgb_to_8_bit(arr).astype(np.uint8)
        argmax = Image.fromarray(eight_bit)
        rgb = (eight_bit > 0).astype(int)
        rgb = (rgb * 255).astype(np.uint8)
        argmax_v = Image.fromarray(rgb)
        argmax.save(f"{save_folder_path1}\{file.stem}.png")
        argmax_v.save(f"{save_folder_path2}\{file.stem}.png")

if __name__ == "__main__":
    predictions_argmax_from_folder_binary(r"C:\Users\magfa\Documents\Master\Masteroppgave\experiments\two-step model\phase extraction run 1\predictions",
                                   r"C:\Users\magfa\Documents\Master\Masteroppgave\experiments\two-step model\phase extraction run 1\predictions_argmax",
                                   r"C:\Users\magfa\Documents\Master\Masteroppgave\experiments\two-step model\phase extraction run 1\predictions_argmax_v")
        