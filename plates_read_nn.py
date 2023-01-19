from keras import layers
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model, Sequential
import numpy as np
import tensorflow as tf
from plates_generate import create_plates
from train_model_characters import load_images_dict_from_folder
import cv2


# def get_model(img_side_size):
#     # Build the U-Net model
#     input_img = Input((img_side_size, img_side_size, 3))
#
#     c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
#     c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
#     p1 = MaxPooling2D((2, 2))(c1)
#
#     c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
#     c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
#     p2 = MaxPooling2D((2, 2))(c2)
#
#     c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
#     c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
#     p3 = MaxPooling2D((2, 2))(c3)
#
#     c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
#     c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
#     p4 = MaxPooling2D((2, 2))(c4)
#
#     c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
#     c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
#
#     u6 = UpSampling2D((2, 2))(c5)
#     u6 = concatenate([u6, c4])
#     c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
#     c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
#
#     u7 = UpSampling2D((2, 2))(c6)
#     u7 = concatenate([u7, c3])
#     c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
#     c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
#
#     u8 = UpSampling2D((2, 2))(c7)
#     u8 = concatenate([u8, c2])
#     c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
#     c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
#
#     u9 = UpSampling2D((2, 2))(c8)
#     u9 = concatenate([u9, c1], axis=3)
#     c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
#     c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
#
#     outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
#     return Model(inputs=[input_img], outputs=[outputs])
#
#
# # Build model
# model = get_model(img_side_size)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
# model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
#

def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x


def build_unet_model(img_side_size):
    # inputs
    inputs = layers.Input(shape=(img_side_size, img_side_size, 1))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    outputs = layers.Conv2D(3, 1, padding="same", activation="softmax")(u9)
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model


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


# Define a function to compute the forward and backward pass
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss_value = loss_fn(labels, logits)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    mean_iou(labels, logits)
    return loss_value


img_side_size = 140
create_plates(10000)

plates_dict = load_images_dict_from_folder("plates")
plates = list(plates_dict.values())
np.reshape(plates, (img_side_size, img_side_size, 1))

masks_dict = load_images_dict_from_folder("masks")
masks = list(masks_dict.values())
np.reshape(masks, (img_side_size, img_side_size, 1))

# Load and preprocess the images and masks
train_dataset = tf.data.Dataset.from_tensor_slices((plates, masks))

BATCH_SIZE = 64
BUFFER_SIZE = 256
train_batches = train_dataset.skip(1010).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_batches = train_dataset.take(1000).batch(BATCH_SIZE)
test_dataset = train_dataset.skip(1000).take(10)

# Build the U-Net model
model = build_unet_model(img_side_size)
model.summary()
tf.keras.utils.plot_model(
    model,
    to_file="unet_model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=True,
)

# Define a loss function and an optimizer
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Use the `tf.keras.metrics.MeanIoU` metric to track the mean IoU during training
mean_iou = tf.keras.metrics.MeanIoU(num_classes=2)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_dataset:
        loss_value = train_step(inputs, labels)
    print("Epoch {}: Loss: {:.4f} Mean IoU: {:.4f}".format(epoch, loss_value, mean_iou.result()))

NUM_EPOCHS = 20
TRAIN_LENGTH = len(plates_dict) - 2000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
model_history = model.fit(train_batches,
                          epochs=NUM_EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=validation_batches)

show_predictions(test_dataset)
