import numpy as np

def rgb_to_8_bit(three_channel_tensor, axis = 1):
    pred_array = np.asarray(three_channel_tensor)
    
    return np.argmax(pred_array, axis=axis)