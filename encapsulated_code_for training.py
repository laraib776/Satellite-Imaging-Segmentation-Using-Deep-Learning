# Install necessary libraries
# !pip install patchify segmentation-models
# !pip install tensorflow keras segmentation-models
# !pip install seaborn

import os
import cv2
import numpy as np
from PIL import Image
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler
import random
import segmentation_models as sm

from keras.utils import get_custom_objects
import tensorflow.keras as keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from keras.callbacks import Callback
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns

# Print the current working directory
print(os.getcwd())
# Path to the 'DubiaDataset' folder
path = './DubiaDataset'

# Ensure the path exists
if os.path.exists(path):
    print(f"Contents of the directory {path}:")
    for root, dirs, files in os.walk(path):
        for name in files:
            print(os.path.join(root, name))
        for name in dirs:
            print(os.path.join(root, name))
else:
    print("The specified path does not exist.")

# Initialize the scaler
minmaxscaler = MinMaxScaler()

# Define dataset paths
import os

dataset_root_folder = '.'
dataset_name = "DubiaDataset"

# Patch size for images
image_patch_size = 256

# Initialize datasets
image_dataset = []
mask_dataset = []

# Helper function to preprocess images
def preprocess_images(image_type, image_extension):
    for tile_id in range(1, 8):
        for image_id in range(1, 20):
            image_path = f'{dataset_root_folder}/{dataset_name}/Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_extension}'
            image = cv2.imread(image_path, 1)

            if image is not None:
                if image_type == 'masks':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                size_x = (image.shape[1] // image_patch_size) * image_patch_size
                size_y = (image.shape[0] // image_patch_size) * image_patch_size

                image = Image.fromarray(image)
                image = image.crop((0, 0, size_x, size_y))
                image = np.array(image)
                patched_images = patchify(image, (image_patch_size, image_patch_size, 3), step=image_patch_size)

                for i in range(patched_images.shape[0]):
                    for j in range(patched_images.shape[1]):
                        if image_type == 'images':
                            individual_patched_image = patched_images[i, j, :, :]
                            individual_patched_image = minmaxscaler.fit_transform(individual_patched_image.reshape(-1, individual_patched_image.shape[-1])).reshape(individual_patched_image.shape)
                            individual_patched_image = individual_patched_image[0]
                            image_dataset.append(individual_patched_image)
                        elif image_type == 'masks':
                            individual_patched_mask = patched_images[i, j, :, :]
                            individual_patched_mask = individual_patched_mask[0]
                            mask_dataset.append(individual_patched_mask)

# Preprocess images and masks
preprocess_images('images', 'jpg')
preprocess_images('masks', 'png')

# Convert datasets to numpy arrays
image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

# Define the class colors
class_colors = {
    'building': '#3C1098',
    'land': '#8429F6',
    'road': '#6EC1E4',
    'vegetation': '#FEDD3A',
    'water': '#E2A929',
    'unlabeled': '#9B9B9B'
}

# Convert hex colors to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return np.array(tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)))

# Define RGB values for each class
rgb_classes = {name: hex_to_rgb(color) for name, color in class_colors.items()}

# Convert RGB to label
def rgb_to_label(label):
    label_segment = np.zeros(label.shape, dtype=np.uint8)
    for i, (class_name, rgb_value) in enumerate(rgb_classes.items()):
        label_segment[np.all(label == rgb_value, axis=-1)] = i
    return label_segment[:, :, 0]

# Convert mask dataset to labels
labels = np.array([rgb_to_label(mask) for mask in mask_dataset])
labels = np.expand_dims(labels, axis=3)

# Convert labels to categorical
total_classes = len(rgb_classes)
labels_categorical_dataset = to_categorical(labels, num_classes=total_classes)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_categorical_dataset, test_size=0.2, random_state=100)

# Model definition
def multi_unet_model(n_classes, image_height, image_width, image_channels):
    inputs = Input((image_height, image_width, image_channels))

    c1 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(inputs)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

    outputs = Conv2D(n_classes, (1, 1), activation="softmax")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Define image shape
image_height = image_patch_size
image_width = image_patch_size
image_channels = 3

# Build model
model = multi_unet_model(n_classes=total_classes, image_height=image_height, image_width=image_width, image_channels=image_channels)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
class PlotLearning(Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print(f"End epoch {epoch} of training; got log keys: {keys}")

# Train the model  

history = model.fit(X_train, y_train, 
                    batch_size=16, 
                    epochs=100, 
                    validation_data=(X_test, y_test), )
                    # callbacks=[PlotLearning()]

# Evaluate the model
model.evaluate(X_test, y_test)

# Visualize model architecture
# plot_model(model, show_shapes=True)

# Save the model
model.save('unet_model.h5')



