import cv2
from sklearn.neighbors import KNeighborsClassifier
import cv2
import os
import numpy as np
from skimage.feature import hog
from utils import load_bw_images_dict_from_folder
from skimage.metrics import structural_similarity as ssim
import random


def extract_features(img):
    return hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), visualize=False,
               feature_vector=True)


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images;
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # the lower MSE, the more "similar" the two images are
    return err


# label_weight defines, how rare is the character. The higher weight - the more complicated or rare is character.
# For example, weight of I is 1, weight of P, L - 2, weight of R, F - 3
'''Come up with better way to add weights into formula'''


def similarity_with_character(label, label_weight, db_images, image):
    # varies in [0, inf], where 0 is perfectly same images
    min_mse = 10000000
    # varies in [-1, 1], where 1 is perfectly same images
    max_ssim = 0
    N = len(db_images)
    for db_image in db_images:
        min_mse = min(min_mse, mse(db_image, image))
        max_ssim = max(max_ssim, ssim(db_image, image))
    return min_mse, max_ssim


def get_character_weight(character):
    character_weights_dict = {
        "A": 2,
        "B": 3,
        "C": 1,
        "D": 3,
        "E": 4,
        "F": 3,
        "G": 2,
        "H": 1,
        "I": 1,
        "J": 2,
        "K": 3,
        "L": 2,
        "M": 4,
        "N": 2,
        "O": 3,
        "P": 3,
        "Q": 4,
        "R": 4,
        "S": 2,
        "T": 2,
        "U": 1,
        "V": 1,
        "W": 2,
        "X": 2,
        "Y": 1,
        "Z": 4,
        "-": 1,
        "1": 2,
        "2": 3,
        "3": 2,
        "4": 3,
        "5": 2,
        "6": 3,
        "7": 3,
        "8": 4,
        "9": 3,
        "0": 3,
    }
    return character_weights_dict[str(character)]


def create_and_train_model():
    chars_dict = load_bw_images_dict_from_folder("database/characters")
    chars_dict_values = np.reshape(list(chars_dict.values()), (len(chars_dict), 15, 20))
    features = np.reshape(list(chars_dict.values()),
                          (len(chars_dict), 15 * 20))  # [extract_features(img) for img in chars_dict_values]

    labels_keys = list(chars_dict.keys())
    labels = []
    for item in labels_keys:
        if item[0] == "a":
            labels.append("-")
        else:
            labels.append(item[0])

    # Train the knn model using the training sets
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(features, labels)

    return model


def convert_to_text(img_list):
    img_list = np.reshape(img_list, (len(img_list), 15, 20))
    # Try pixel values as features instead of hog
    test_features = img_list  # [extract_features(img) for img in img_list]

    chars_dict = load_bw_images_dict_from_folder("database/characters")
    chars_dict_values = np.reshape(list(chars_dict.values()), (len(chars_dict), 15, 20))
    chars_features = chars_dict_values  # [extract_features(img) for img in chars_dict_values]

    labels_keys = list(chars_dict.keys())
    labels = []
    labelled_chars = {}
    for i, item in enumerate(labels_keys):
        if item[0] == "A" and item[1] == "A":
            if "-" not in labelled_chars:
                labelled_chars["-"] = []
            labelled_chars["-"].append(chars_features[i])
        else:
            if item[0] not in labelled_chars:
                labelled_chars[item[0]] = []
            labelled_chars[item[0]].append(chars_features[i])

    predicted_text = []
    for i, img in enumerate(img_list):
        # cv2.imshow("1", np.uint8(img))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        predicted_char = ""
        min_mse = 100000
        max_ssim = -1
        deviation_ssim = 0.3
        predicted_char_weight = 0
        chars_to_check = list(labelled_chars.keys())
        if i == 0:
            chars_to_check = [char for char in chars_to_check if not char.isdigit()]
            chars_to_check.remove("-")
        if i == len(img_list) - 1:
            chars_to_check = [char for char in chars_to_check if char.isdigit()]

        for char in chars_to_check:
            labelled_chars_images = labelled_chars[char]
            char_weight = get_character_weight(char)
            mse, ssim = similarity_with_character(char, char_weight, labelled_chars_images, img)
            if (mse <= min_mse and ssim >= max_ssim):
                min_mse = mse
                max_ssim = ssim
                predicted_char = char
        predicted_text.append(predicted_char)
#        print("Predicted char: ", max_ssim, min_mse, predicted_char)
    return predicted_text
