import cv2
from sklearn.neighbors import KNeighborsClassifier
import cv2
import os
import numpy as np
from skimage.feature import hog


def extract_features(img):
    return hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), visualize=False,
               feature_vector=True)


def load_images_dict_from_folder(folder):
    chars_dict = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if img is not None:
            chars_dict[filename] = img
    return chars_dict


def create_and_train_model():
    model = KNeighborsClassifier(n_neighbors=3)

    chars_dict = load_images_dict_from_folder("characters")
    chars_dict_values = np.reshape(list(chars_dict.values()), (len(chars_dict), 15, 20))

    # Try pixel values as features instead of hog (results haven't improved)
    features = np.reshape(list(chars_dict.values()), (len(chars_dict), 15*20)) #[extract_features(img) for img in chars_dict_values]

    labels_keys = list(chars_dict.keys())
    labels = []
    for item in labels_keys:
        if item[0] == "a":
            labels.append("-")
        else:
            labels.append(item[0])

    # Train the model using the training sets
    model.fit(features, labels)

    return model


def convert_to_text(model, img_list):
    img_list = np.reshape(img_list, (len(img_list), 15, 20))
    # Try pixel values as features instead of hog (results haven't improved)
    test_features = np.reshape(img_list, (len(img_list), 15*20)) # [extract_features(img) for img in img_list]
    return model.predict(test_features)
