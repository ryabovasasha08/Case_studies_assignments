import numpy as np
import tensorflow as tf
from generate.plates_generate import create_plates
from utils import load_images_dict_from_folder, load_bw_images_dict_from_folder
import cv2
from PIL import Image
from unet_model import build_unet_model
from sklearn.model_selection import train_test_split

ORIGINAL_IMG_SIZE = 256

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


# this will be needed for pretty pic of the model for presentation/report
def plot_model(model):
    tf.keras.utils.plot_model(model, to_file="unet_model.png", show_shapes=True, show_dtype=False,
                              show_layer_names=True, rankdir="TB", expand_nested=False,
                              dpi=96, layer_range=None, show_layer_activations=True, )


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def show_predictions(X_val, Y_val, num=1):
    # result of predictions has shape (256, 256, 3) and contains values from 0 to 1
    pred_masks = model.predict(X_val)
    for i, image in enumerate(X_val):
        mask = Y_val[i]
        pred_mask = pred_masks[i]
        display(image, mask, pred_mask)
        if i > 5:
            break


# If the plates and masks folders are empty or don't contain enough elements, clear them and run this line
# create_plates(10000)

# Load and preprocess the images and masks
plates_dict = load_images_dict_from_folder("database/plates")
plates_raw = list(plates_dict.values())
plates_raw = np.reshape(plates_raw, (len(plates_raw), ORIGINAL_IMG_SIZE, ORIGINAL_IMG_SIZE, 3))

# plates = [tf.image.resize(plate, (IMG_SIZE, IMG_SIZE), method="nearest") for plate in plates] - no need, since images are generated 256*256 now

masks_dict = load_images_dict_from_folder("database/masks")
masks_raw = list(masks_dict.values())
masks_raw = np.reshape(masks_raw, (len(masks_raw), ORIGINAL_IMG_SIZE, ORIGINAL_IMG_SIZE, 3))
# masks = [tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest") for mask in masks] - no need, since images are generated 256*256


masks = []
plates = []
for i in range(0,len(masks_raw)):
    plate = cv2.resize(plates_raw[i], dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
    plate = plate/255
    plates.append(plate)

    mask = cv2.resize(masks_raw[i], dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
    mask = mask/255
    masks.append(mask)

# To check the mask and plate eye test:
# cv2.imshow("Plate", np.uint8(plates[0]))
# cv2.imshow("Mask", np.uint8(masks[0]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Create dataset and split it into train, validation and test + split into batches
X_train, X_val, Y_train, Y_val = train_test_split(plates, masks, random_state=104, test_size=0.3, shuffle=True)

# Build the U-Net model
model = build_unet_model(ORIGINAL_IMG_SIZE)
model.summary()
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

NUM_EPOCHS = 20
BATCH_SIZE = 64
model_history = model.fit(X_train, Y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, Y_val),
                          callbacks=[DisplayCallback()])

model.save("read_nn/unet")
print("model saved: read_nn/unet")

oss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

# After you created and trained the model, it will get saved in read_nn/unet, so you can just load it from there instead of re-training it every time:
# model = tf.keras.models.load_model('read_nn/unet')

show_predictions(X_val, Y_val)
