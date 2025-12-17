# melanoma_detection.py

"""
CNN-based Melanoma Detection
Detects melanoma from skin images using a Convolutional Neural Network (CNN).
"""

# Import required libraries
import pathlib
import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img

# Optional: Augmentation
import Augmentor

# -----------------------------
# Define dataset paths
# -----------------------------
data_dir_train = pathlib.Path("path_to_train_dataset/Train/")
data_dir_test = pathlib.Path("path_to_test_dataset/Test/")

# -----------------------------
# Count images in train and test directories
# -----------------------------
image_count_train = len(list(data_dir_train.glob('*/*.jpg')))
image_count_test = len(list(data_dir_test.glob('*/*.jpg')))
print(f"Number of training images: {image_count_train}")
print(f"Number of testing images: {image_count_test}")

# -----------------------------
# Data Visualization
# -----------------------------
image_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    batch_size=32,
    image_size=(180, 180),
    label_mode='categorical',
    seed=123
)

class_names = image_dataset.class_names
print(f"Classes: {class_names}")

# Store file paths per class
files_path_dict = {}
for c in class_names:
    files_path_dict[c] = list(map(lambda x: str(data_dir_train) + '/' + c + '/' + x, os.listdir(str(data_dir_train) + '/' + c)))

# Visualize one image per class
plt.figure(figsize=(15, 15))
for index, c in enumerate(class_names):
    path_list = files_path_dict[c][:1]
    plt.subplot(3, 3, index + 1)
    plt.imshow(load_img(path_list[0], target_size=(180, 180)))
    plt.title(c)
    plt.axis("off")
plt.show()

# -----------------------------
# Class distribution
# -----------------------------
def class_distribution_count(directory):
    count = []
    for path in pathlib.Path(directory).iterdir():
        if path.is_dir():
            count.append(len([name for name in os.listdir(path)
                              if os.path.isfile(os.path.join(path, name))]))
    sub_directory = [name for name in os.listdir(directory)
                     if os.path.isdir(os.path.join(directory, name))]
    return pd.DataFrame(list(zip(sub_directory, count)), columns=['Class', 'No. of Image'])

df = class_distribution_count(data_dir_train)
print(df)

plt.figure(figsize=(10, 8))
sns.barplot(x="No. of Image", y="Class", data=df, label="Class")
plt.show()

# -----------------------------
# Data Augmentation using Augmentor
# -----------------------------
for i in class_names:
    p = Augmentor.Pipeline(str(data_dir_train / i))
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.sample(500)  # Add 500 samples per class

# Count total images after augmentation
image_count_train = len(list(data_dir_train.glob('*/output/*.jpg')))
print(f"Number of training images after augmentation: {image_count_train}")

# -----------------------------
# Prepare training and validation datasets
# -----------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    batch_size=32,
    image_size=(180, 180),
    label_mode='categorical',
    seed=123,
    subset="training",
    validation_split=0.2
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    batch_size=32,
    image_size=(180, 180),
    label_mode='categorical',
    seed=123,
    subset="validation",
    validation_split=0.2
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# Build CNN model
# -----------------------------
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(len(class_names), activation='softmax')
])

model.summary()

# -----------------------------
# Compile model
# -----------------------------
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint("model.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
earlystop = EarlyStopping(monitor="val_accuracy", patience=5, mode="auto", verbose=1)

# -----------------------------
# Train the model
# -----------------------------
epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint, earlystop])

# -----------------------------
# Plot training curves
# -----------------------------
epochs_range = range(earlystop.stopped_epoch + 1)

plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# -----------------------------
# Test prediction example
# -----------------------------
test_image_path = os.path.join(data_dir_test, class_names[1], '*')
test_image_file = glob(test_image_path)[-1]
test_image = load_img(test_image_file, target_size=(180, 180, 3))

plt.imshow(test_image)
plt.grid(False)
plt.show()

img = np.expand_dims(test_image, axis=0)
pred = model.predict(img)
pred_class = class_names[np.argmax(pred)]

print(f"Actual Class: {class_names[1]}")
print(f"Predicted Class: {pred_class}")

