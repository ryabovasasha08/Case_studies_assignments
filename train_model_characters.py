import cv2
from sklearn.neighbors import KNeighborsClassifier
import cv2
import os
import numpy as np
from skimage.feature import hog


def load_chars_dict_from_folder(folder):
    chars_dict = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if img is not None:
            chars_dict[filename] = img
    return chars_dict


model = KNeighborsClassifier(n_neighbors=3)

chars_dict = load_chars_dict_from_folder("characters")
features = [hog(img) for img in chars_dict.values()]

labels = list(chars_dict.keys())
labels = [item[0] for item in labels]

# Train the model using the training sets
model.fit(features, labels)

print("wait")
# Predict Output
predicted = model.predict(features[0])  # 0:Overcast, 2:Mild
print(predicted)
print(predicted)
