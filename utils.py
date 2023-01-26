import os
import cv2
import numpy as np

def load_images_dict_from_folder(folder):
    chars_dict = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if img is not None:
            chars_dict[filename] = img
    return chars_dict
