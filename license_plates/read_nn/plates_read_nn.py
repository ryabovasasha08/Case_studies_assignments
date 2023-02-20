import numpy as np
import tensorflow as tf
from generate.plates_generate import create_plates
from utils import load_images_dict_from_folder, load_bw_images_dict_from_folder
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from unet_model import build_unet_model
from sklearn.model_selection import train_test_split

ORIGINAL_IMG_SIZE = 256

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint['plate'], (75, 128))
    input_mask = tf.image.resize(datapoint['mask'], (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def display(display_list, label=""):
    if label != "":
        plt.figure(label)
    else:
        plt.figure()
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def show_predictions(label="", dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))], label)


# If the plates and masks folders are empty or don't contain enough elements, clear them and run this line
# create_plates(500)

# Load and preprocess the images and masks
plates_dict = load_images_dict_from_folder("database/plates")
plates = list(plates_dict.values())
plates = np.reshape(plates, (len(plates), ORIGINAL_IMG_SIZE, ORIGINAL_IMG_SIZE, 3))

masks_dict = load_images_dict_from_folder("database/masks")
masks = list(masks_dict.values())

masks = np.reshape(masks, (len(masks), ORIGINAL_IMG_SIZE, ORIGINAL_IMG_SIZE, 3))
for i in range(0, len(masks)):
    masks[i] = np.reshape(masks[i][:, :, 0], (ORIGINAL_IMG_SIZE, ORIGINAL_IMG_SIZE, 1))

# To check the mask and plate eye test:
# cv2.imshow("Plate", np.uint8(plates[0]))
# cv2.imshow("Mask", np.uint8(masks[0]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Create dataset and split it into train, validation and test + split into batches

tf_dataset = tf.data.Dataset.from_tensor_slices({'plate': plates, 'mask': masks})

images = tf_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

BATCH_SIZE = 64
BUFFER_SIZE = 1000

batches = images.batch(BATCH_SIZE)

# After you created and trained the model, it will get saved in read_nn/unet, so you can just load it from there instead of re-training it every time:
model = tf.keras.models.load_model('read_nn/unet')

#Show predictions for first element of dataset
show_predictions(dataset=batches, num=1)
