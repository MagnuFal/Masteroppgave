import numpy as np
from pathlib import Path
from PIL import Image

def rgb_to_8_bit(three_channel_tensor, axis = 1):
    pred_array = np.asarray(three_channel_tensor)
    
    return np.argmax(pred_array, axis=axis)

if __name__ == "__main__":
    path = r"C:\Users\magfa\Documents\Master\Masteroppgave\experiments\dataset_3\dataset_3_improved_first_run\dataset_3_improved_first_run_predictions"
    folder = Path(path)
    for file in folder.iterdir():
        img = Image.open(file)
        arr = np.asarray(img)
        