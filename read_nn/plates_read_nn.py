import numpy as np
import tensorflow as tf
from generate.plates_generate import create_plates
from utils import load_images_dict_from_folder
import cv2

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset, num=1):
    i = 0
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        # concatenate image Horizontally
        img3 = np.concatenate((image[0], mask[0], create_mask(pred_mask)), axis=1)
        cv2.imshow(img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        i += 1
        if i > 5:
            break


#create_plates(1000)

# Load and preprocess the images and masks
plates_dict = load_images_dict_from_folder("dataset/plates")
plates = list(plates_dict.values())
plates = np.reshape(plates, (len(plates), img_side_size, img_side_size, 1))
# plates = [tf.image.resize(plate, (IMG_SIZE, IMG_SIZE), method="nearest") for plate in plates] - no need, since images are generated 256*256 now

masks_dict = load_images_dict_from_folder("dataset/masks")
masks = list(masks_dict.values())
masks = np.reshape(masks, (len(masks), img_side_size, img_side_size, 1))
# masks = [tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest") for mask in masks] - no need, since images are generated 256*256

#Create dataset and split it into train, validation and test + split into batches
train_dataset = tf.data.Dataset.from_tensor_slices((plates, masks))
BATCH_SIZE = 64
BUFFER_SIZE = 256
train_batches = train_dataset.skip(1010).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_batches = train_dataset.take(1000).batch(BATCH_SIZE)
test_dataset = train_dataset.skip(1000).take(10)

# Build the U-Net model
model = build_unet_model(IMG_SIZE)
model.summary()

# this is needed for pretty pic of the model for presentation/report
# tf.keras.utils.plot_model(
#     model,
#     to_file="unet_model.png",
#     show_shapes=True,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=True,
# )

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    for inputs, labels in train_dataset:
        loss_value = train_step(inputs, labels)
    print("Epoch {}: Loss: {:.4f}".format(epoch, loss_value))

NUM_EPOCHS = 20
TRAIN_LENGTH = len(plates_dict) - 1010 #all set - test set - validation set
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
model_history = model.fit(train_batches,
                          epochs=NUM_EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=validation_batches)

show_predictions(test_dataset)
