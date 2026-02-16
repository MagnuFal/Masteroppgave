import numpy as np
from PIL import Image
from .to_RGB import to_rgb
from pathlib import Path

def raw_image(needle_arr, script_arr):

    needle_arr = needle_arr // needle_arr.max()
    script_arr= script_arr // script_arr.max()
    raw = needle_arr + script_arr

    return (raw > 0).astype(int)

def label_image(needle_arr, script_arr):

    needle_arr = needle_arr // needle_arr.max()
    script_arr= script_arr // script_arr.max()

    script_arr = script_arr * 2

    label = script_arr + needle_arr
    label_overlap = (label == 3)
    label = label - label_overlap
    label_overlap = (label_overlap // 3) * 2

    return label + label_overlap

def path_to_arr(path):
    img = Image.open(path)
    return np.asarray(img)

def raw_and_label_from_folder(script_folder_path, needle_folder_path,
                              raw_folder_path, label_folder_path,
                              v_raw_folder_path, v_label_folder_path):
    script_folder = Path(script_folder_path)
    needle_folder = Path(needle_folder_path)

    needle_lst = []
    script_lst = []

    for file in script_folder.iterdir():
        script_lst.append(path_to_arr(file))

    for file in needle_folder.iterdir():
        needle_lst.append(path_to_arr(file))

    for i in range(len(needle_lst)):
        needle_arr = needle_lst[i].astype(int)
        count_arr = np.unique(needle_arr, return_counts=True)[1]
        if count_arr[0] < count_arr[1]:
            needle_arr_bool = needle_arr.astype(bool)
            needle_arr_bool_inverted = np.invert(needle_arr_bool)
            needle_arr = needle_arr_bool_inverted.astype(int)
        raw = raw_image(needle_arr, script_lst[i]).astype(np.uint8)
        label = label_image(needle_arr, script_lst[i]).astype(np.uint8)
        print(np.unique(raw, return_counts=True))
        print(np.unique(label, return_counts=True))
        raw_img = Image.fromarray(raw)
        label_img = Image.fromarray(label)
        v_raw_img = Image.fromarray((raw * 255).astype(np.uint8))
        v_label_img = Image.fromarray(to_rgb(label).astype(np.uint8))

        raw_img.save(f"{raw_folder_path}\{i}.png")
        label_img.save(f"{label_folder_path}\{i}.png")
        v_raw_img.save(f"{v_raw_folder_path}\{i}.png")
        v_label_img.save(f"{v_label_folder_path}\{i}.png")
        
        
        # Utvide funksjonen sånn at den lager versjoner av bildene som
        # kan sees - Gange med 255 for raw og to_rgb for label

script = r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\script and needles\script"
needle = r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\script and needles\needles"
raw = r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\script and needles\raw"
label = r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\script and needles\label"
v_raw = r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\script and needles\v_raw"
v_label = r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\script and needles\v_label"

raw_and_label_from_folder(script, needle, raw, label, v_raw, v_label)