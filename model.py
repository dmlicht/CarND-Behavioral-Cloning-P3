from typing import Iterable, List

import pandas as pd
import numpy as np
import os

import cv2
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, MaxPooling2D, Cropping2D
from keras.models import Sequential

TRAINING_DIR = 'training2'
IMG_DIR = 'IMG'
DRIVE_LOG_PATH = os.path.join(".", TRAINING_DIR, "driving_log.csv")
SIDE_IMAGE_CORRECTION = 1
MODEL_FILE = 'model.h5'


def augment_flip(X, y):
    flipped_images = [cv2.flip(image, 1) for image in X]
    flipped_angles = [-1 * angle for angle in y]
    return np.array(flipped_images), np.array(flipped_angles)


def main():
    drive_log = pd.read_csv(DRIVE_LOG_PATH, names=["center", "left", "right", "angle", "throttle", "break", "speed"])

    X_train, y_train = load_center_data(drive_log)
    # X_train, y_train = load_multi_data(drive_log)

    X_aug, y_aug = augment_flip(X_train, y_train)

    X_train = np.concatenate((X_train, X_aug))
    y_train = np.concatenate((y_train, y_aug))

    # pprint(X_train[0].shape) # -> (160, 320, 3)
    model = lenet()

    checkpoint = ModelCheckpoint(MODEL_FILE, monitor='val_loss', verbose=1, save_best_only=True)
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, callbacks=[checkpoint])
    # model.save("model.h5")


def load_center_data(drive_log):
    """ Load images and angles for the center Camera. """
    center_image_paths = drive_log["center"]
    local_img_paths = convert_to_local_paths(center_image_paths)
    return load_images(local_img_paths), drive_log["angle"]


def convert_to_local_paths(paths: List[str]) -> List[str]:
    return [convert_to_local_path(path) for path in paths]


def convert_to_local_path(path: str) -> str:
    tokens = path.split('/')
    filename = tokens[-1]
    return os.path.join(".", TRAINING_DIR, IMG_DIR, filename)


def load_multi_data(drive_log, side_correction=1):
    """ Load images and angles for the center and side cameras with augmented angles for the side cameras. """
    center_image_paths = convert_to_local_paths(drive_log["center"])
    left_image_paths = convert_to_local_paths(drive_log["left"])
    right_image_paths = convert_to_local_paths(drive_log["right"])
    angle = drive_log["angle"]

    center_images = load_images(center_image_paths)
    left_images = load_images(left_image_paths)
    right_images = load_images(right_image_paths)

    X_train = np.concatenate((center_images, left_images, right_images))
    y_train = np.concatenate((angle, angle + side_correction, angle - side_correction))
    return X_train, y_train


def load_images(paths: Iterable[str]) -> np.array:
    """ Load a collection of images from paths. """
    return np.array([cv2.imread(path) for path in paths])


def linear():
    """ Prove we can make a model and run it properly """
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(_normalize_pixel))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def lenet():
    """ Basic implementation of LeNet architecture """
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(_normalize_pixel))
    model.add(Conv2D(8, 4, 4, activation="relu", border_mode="valid"))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, 4, 4, activation="relu", border_mode="valid"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Dense(120, activation="relu"))
    model.add(Dropout(.5))
    model.add(Dense(84, activation="relu"))
    model.add(Dropout(.5))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def _normalize_pixel(x):
    """ Scale pixel to 0-1 and center around 0."""
    return (x / 255.0) - 0.5


if __name__ == '__main__':
    main()
