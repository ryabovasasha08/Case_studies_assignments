import os
import cv2
import numpy as np


def load_bw_images_dict_from_folder(folder):
    chars_dict = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename)).astype(np.float32)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            otsu_threshold, img = cv2.threshold(np.uint8(img), 0, 255, cv2.THRESH_OTSU)
            chars_dict[filename] = img
    return chars_dict

def load_images_dict_from_folder(folder):
    chars_dict = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename)).astype(np.float32)
        if img is not None:
            chars_dict[filename] = img
    return chars_dict
