import numpy as np
import tensorflow as tf
from generate.plates_generate import create_plates
from utils import load_images_dict_from_folder, load_bw_images_dict_from_folder
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from unet_model import build_unet_model
from sklearn.model_selection import train_test_split

ORIGINAL_IMG_SIZE = 256


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            label = "Sample Prediction after epoch " + str(epoch)
            show_predictions(label=label)


# this will be needed for pretty pic of the model for presentation/report
def plot_model(model):
    tf.keras.utils.plot_model(model, to_file="unet_model.png", show_shapes=True, show_dtype=False,
                              show_layer_names=True, rankdir="TB", expand_nested=False,
                              dpi=96, layer_range=None, show_layer_activations=True, )


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint['plate'], (128, 128))
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
    plt.show(block=False)


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

# plates = [tf.image.resize(plate, (IMG_SIZE, IMG_SIZE), method="nearest") for plate in plates] - no need, since images are generated 256*256 now

masks_dict = load_images_dict_from_folder("database/masks")
masks = list(masks_dict.values())
masks = np.reshape(masks, (len(masks), ORIGINAL_IMG_SIZE, ORIGINAL_IMG_SIZE, 3))
# masks = [tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest") for mask in masks] - no need, since images are generated 256*256

# To check the mask and plate eye test:
# cv2.imshow("Plate", np.uint8(plates[0]))
# cv2.imshow("Mask", np.uint8(masks[0]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Create dataset and split it into train, validation and test + split into batches

test_y = []
test_x = []
train_x = []
train_y = []

TRAIN_TEST_SPLIT = 0.7

for i in range(0, int(len(plates) * TRAIN_TEST_SPLIT)):
    train_x.append(plates[i])
    train_y.append(np.reshape(masks[i][:, :, 0], (ORIGINAL_IMG_SIZE, ORIGINAL_IMG_SIZE, 1)))
for i in range(int(len(plates) * TRAIN_TEST_SPLIT) + 1, len(plates)):
    test_x.append(plates[i])
    test_y.append(np.reshape(masks[i][:, :, 0], (ORIGINAL_IMG_SIZE, ORIGINAL_IMG_SIZE, 1)))

train_dataset = tf.data.Dataset.from_tensor_slices({'plate': train_x, 'mask': train_y})
test_dataset = tf.data.Dataset.from_tensor_slices({'plate': test_x, 'mask': test_y})

train_images = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = test_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = len(train_x) // BATCH_SIZE

train_batches = (
    train_images.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().prefetch(buffer_size=tf.data.AUTOTUNE))
test_batches = test_images.batch(BATCH_SIZE)

# check that display works correctly:
for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])

EPOCHS = 10

model = build_unet_model()
model.summary()
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=test_batches,
                          callbacks=[DisplayCallback()])

model.save("read_nn/unet")
print("model saved: read_nn/unet")

# After you created and trained the model, it will get saved in read_nn/unet, so you can just load it from there instead of re-training it every time:
# model = tf.keras.models.load_model('read_nn/unet')
