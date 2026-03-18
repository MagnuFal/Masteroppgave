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
    needle_max = needle_arr.max()
    script_max = script_arr.max()
    if needle_max != 0:
        needle_arr = needle_arr // needle_arr.max()
    if script_max != 0:
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

def revert_if_inverted(arr):
    arr_max = arr.max()
    if arr_max != 0:
        arr = arr // arr_max
    unique_arr, count_arr = np.unique(arr, return_counts=True)[0], np.unique(arr, return_counts=True)[1]
    if len(count_arr) == 1:
         if unique_arr[0] == 1:
            arr_bool = arr.astype(bool)
            arr_bool_inverted = np.invert(arr_bool)
            arr = arr_bool_inverted.astype(int)
    elif count_arr[0] < count_arr[1]:
        arr_bool = arr.astype(bool)
        arr_bool_inverted = np.invert(arr_bool)
        arr = arr_bool_inverted.astype(int)
    return arr

def add_to_stem_of_file_path(file_path, str = None, str2 = None):
    file = Path(file_path)
    absolute_file = file.resolve()
    return absolute_file.with_stem(str2 + absolute_file.stem + str)

def add_same_str_to_stem_in_folder(folder_path, str = None, str2 = None):
    folder = Path(folder_path)
    for file in folder.iterdir():
        modified_path = add_to_stem_of_file_path(file.resolve(), str, str2)
        file.rename(modified_path)

def convert_folder_from_tif_to_png(folder_path, save_folder_path):
    folder = Path(folder_path)
    for file in folder.iterdir():
        img = Image.open(file)
        img.save(f"{save_folder_path}\{file.stem}.png")

def raw_and_label_from_folder(script_folder_path, needle_folder_path,
                               label_folder_path, v_label_folder_path,
                               raw_folder_path = None, v_raw_folder_path = None):
    script_folder = Path(script_folder_path)
    needle_folder = Path(needle_folder_path)

    script_files = {p.stem : p for p in script_folder.iterdir()}
    needle_files = {p.stem : p for p in needle_folder.iterdir()}

    for stem, path in script_files.items():
        script_arr = path_to_arr(path)
        needle_arr = path_to_arr(needle_files[stem])
        needle_arr, script_arr = revert_if_inverted(needle_arr.astype(np.uint8)), revert_if_inverted(script_arr.astype(np.uint8))

        label = label_image(needle_arr, script_arr).astype(np.uint8)
        label_img = Image.fromarray(label)
        v_label_img = Image.fromarray(to_rgb(label).astype(np.uint8))
        label_img.save(f"{label_folder_path}\{stem}.png")
        v_label_img.save(f"{v_label_folder_path}\{stem}.png")

        if (raw_folder_path != None) and (v_raw_folder_path != None):
            raw = raw_image(needle_arr, script_arr).astype(np.uint8)
            raw_img = Image.fromarray(raw)
            v_raw_img = Image.fromarray((raw * 255).astype(np.uint8))
            raw_img.save(f"{raw_folder_path}\{stem}.png")
            v_raw_img.save(f"{v_raw_folder_path}\{stem}.png")


if __name__ == "__main__":
    convert_folder_from_tif_to_png(r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\raw", r"C:\Users\magfa\Documents\Master\Masteroppgave\dataset_generation\raw2")