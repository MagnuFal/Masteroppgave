import numpy as np
from pathlib import Path
from PIL import Image
from to_RGB import to_rgb


## Her er det mye overflødighet -> Lag funksjoner

def rgb_to_8_bit(three_channel_tensor, axis = 2):
    pred_array = np.asarray(three_channel_tensor)
    
    return np.argmax(pred_array, axis=axis)

def confidence_threshold_rgb_to_8_bit(three_channel_array, axis = 2, script_threshold = 0.7, background_threshold = 0.60):
    pred_array = np.asarray(three_channel_array).copy()
    
    #script_pred_array = pred_array[:, :, 2]
#
    #threshold = threshold * 255
#
    #overwrite_array = (script_pred_array > threshold).astype(int)
    #script_array = (overwrite_array * 2).astype(np.uint8)
#
    #argmax = np.argmax(pred_array, axis=axis).astype(np.uint8)
#
    #argmax = argmax - script_array
    #argmax[argmax < 0] = 0
#
    #return (argmax + script_array).astype(np.uint8)

    script_array = pred_array[:, :, 2]
    background_array = pred_array[:, :, 0]

    script_array[script_array > (script_threshold * 255)] = 255
    script_array[background_array > (background_threshold * 255)] = 0

    pred_array[:, :, 2] = script_array

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


def confidence_predictions_argmax_from_folder_rgb(folder_path, save_folder_path1, save_folder_path2):
    folder = Path(folder_path)
    for file in folder.iterdir():
        img = Image.open(file)
        arr = np.asarray(img)
        eight_bit = confidence_threshold_rgb_to_8_bit(arr).astype(np.uint8)
        argmax = Image.fromarray(eight_bit)
        rgb = to_rgb(eight_bit).astype(np.uint8)
        argmax_v = Image.fromarray(rgb)
        argmax.save(f"{save_folder_path1}\{file.stem}.png")
        argmax_v.save(f"{save_folder_path2}\{file.stem}.png")

if __name__ == "__main__":
    folder_path = r"C:\Users\magfa\Documents\Master\Masteroppgave\experiments\2022.09.28_S3400N_PhysMet_IBA_png_run_1\predictions_argmax"
    save_folder_path = r"C:\Users\magfa\Documents\Master\Masteroppgave\experiments\2022.09.28_S3400N_PhysMet_IBA_png_run_1\predictions_argmax_v"
    folder = Path(folder_path)
    save_folder = Path(save_folder_path)

    for file in folder.iterdir():
        img = Image.open(file)
        arr = np.asarray(img)
        print(np.unique(arr, return_counts=True))
        rgb = to_rgb(arr).astype(np.uint8)
        img = Image.fromarray(rgb)
        img.save(save_folder / file.name)
        