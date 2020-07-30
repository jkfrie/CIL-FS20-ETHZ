# -*- coding: utf-8 -*-

import numpy as np
import os

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import concatenate
import keras
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

import util.metrics as metrics
import util.utility as util

## Install the following packages
import natsort
import logging

"""## Constants"""

# Name of the current model
MODEL_NAME = 'fullyCNN_datagenerator_improved_unet_dropout'
IMG_WIDTH = 608
IMG_HEIGHT = 608
EPOCHS = 150
STEPS_PER_EPOCH = 500
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.1
rnd_seed = 10
np.random.seed(rnd_seed)


logging.basicConfig(filename='Logs/messages.log', level=logging.INFO, format="%(asctime)s.%(msecs)03d[%(levelname)s]:%(message)s")
logging.info('Starting ' + MODEL_NAME)

"""## Load Images"""

logging.info('Loading training and test images')
training_image_dir = "training_images/images/"
training_label_dir = "training_images/groundtruth/"

# Load filenames and split original training images, self generated training
# images and test images.
files_image = os.listdir(training_image_dir)
files_image = natsort.natsorted(files_image)
files_image_original = files_image[-100:]
files_image_extra = files_image[:-100]

files_label = os.listdir(training_label_dir)
files_label = natsort.natsorted(files_label)
files_label_original = files_label[-100:]
files_label_extra = files_label[:-100]

files_test = os.listdir(test_image_dir)
files_test = natsort.natsorted(files_test)

# Load Images and labels
training_image_original = util.load_images(training_image_dir, files_image_original, "RGB")
training_image_extra = util.load_images(training_image_dir, files_image_extra, "RGB")
training_label_original = util.load_images(training_label_dir, files_label_original, "L")
training_label_extra = util.load_images(training_label_dir, files_label_extra, "L")

print("TRAINING:")
print(training_image_original.shape)
print(training_label_original.shape)
print(training_image_extra.shape)
print(training_label_extra.shape)

logging.info('Finished loading!')

"""
## Preproess Images
- Training images have size 400x400 and test images have size 608x608. So we
  need to pad training images to same size, for that I use mirror padding for now.
- Get a validation set of untouched original training images.
- Augment original training data with vertical and horizontal flips and 45 degrees
  rotations.
- Also augment validation set to get a better average performance.
- Rescale images
"""
logging.info('Start Preprocessing Phase')

# Mirror padd all training images to get same size as test images
training_image_original = util.padd_images(training_image_original, 608, 608).astype(np.uint8)
training_image_extra = util.padd_images(training_image_extra, 608, 608).astype(np.uint8)
training_label_original = util.padd_images(training_label_original, 608, 608).astype(np.uint8)
training_label_extra = util.padd_images(training_label_extra, 608, 608).astype(np.uint8)

# Create a validation set
training_image_original, validation_image, training_label_original, validation_label = train_test_split(
    training_image_original, training_label_original, test_size=VALIDATION_SPLIT, random_state=rnd_seed)

# Augment original training data and validation set
training_image_original = util.add_flipped_images(training_image_original)
training_label_original = util.add_flipped_images(training_label_original)
validation_image = util.add_flipped_images(validation_image)
validation_label = util.add_flipped_images(validation_label)

training_image_original = util.add_rotated_images(training_image_original)
training_label_original = util.add_rotated_images(training_label_original)
validation_image = util.add_rotated_images(validation_image)
validation_label = util.add_rotated_images(validation_label)

training_image = np.concatenate((training_image_original, training_image_extra), axis=0)
training_label = np.concatenate((training_label_original, training_label_extra), axis=0)
training_label = np.expand_dims(training_label, -1)
validation_label = np.expand_dims(validation_label, -1)

# Rescale validation images/labels and test images because generator will do the same with training data
training_image = training_image.astype(np.float32)/255.0
training_label = training_label.astype(np.float32)/255.0
validation_image = validation_image.astype(np.float32)/255.0
validation_label = validation_label.astype(np.float32)/255.0
logging.info('Finished Preprocessing!')

print(training_image.shape)
print(training_label.shape)
print(validation_image.shape)
print(validation_label.shape)

"""
## Keras Data Generator
We use the Keras Data Generator to augment our training data online while training. This is necessary because of memory consumption.
"""
# We create an instance for the training images, training labels and test images
data_gen_args = dict(width_shift_range=0.1,
                     height_shift_range=0.1,
                     #zoom_range=0.05,
                     #shear_range=0.05,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_generator = image_datagen.flow(
    training_image,
    batch_size=BATCH_SIZE,
    seed=seed)
mask_generator = mask_datagen.flow(
    training_label,
    batch_size=BATCH_SIZE,
    seed=seed)

# Combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

"""## Model: Custom U-Net built in Keras"""

inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))

conv1 = util.convolutional_block(inputs, 16)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = util.convolutional_block(pool1, 32)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = util.convolutional_block(pool2, 64)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = util.convolutional_block(pool3, 128)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = util.convolutional_block(pool4, 256)
pool5 = MaxPooling2D(pool_size=(2,2)) (conv5)

conv6 = util.convolutional_block(pool5, 512)

up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
concat7 = concatenate([up7, conv5])
conv7 = util.convolutional_block(concat7, 256)

up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
concat8 = concatenate([up8, conv4])
conv8 = util.convolutional_block(concat8, 128)

up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
concat9 = concatenate([up9, conv3])
conv9 = util.convolutional_block(concat9, 64)

up10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv9)
concat10 = concatenate([up10, conv2])
conv10 = util.convolutional_block(concat10, 32)

up11 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv10)
concat11 = concatenate([up11, conv1])
conv11 = util.convolutional_block(concat11, 16)
conv12 = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

model = Model(inputs=inputs, outputs=conv12)

"""## Callbacks for Observations"""

model_path = "./Models/{}_model.h5".format(MODEL_NAME)
checkpointer = ModelCheckpoint(model_path,
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)
csv_logger = CSVLogger("./Logs/{}_log.csv".format(MODEL_NAME), separator=',', append=False)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=6,
                               verbose=1,
                               epsilon=1e-4)
early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

"""## Model Training"""

opt = keras.optimizers.Adam(LEARNING_RATE)
opt = keras.optimizers.Nadam(lr=LEARNING_RATE)
logging.info('Compiling Model')
model.compile(
      optimizer=opt,
      loss=metrics.combined_loss,
      metrics=[metrics.iou_coef])

logging.info('Starting Training')
history = model.fit_generator(train_generator,
                              validation_data=(validation_image, validation_label),
                              steps_per_epoch=STEPS_PER_EPOCH,
                              epochs=EPOCHS,
                              callbacks=[checkpointer, csv_logger, lr_reducer, early_stopper])
logging.info('Finished Training')
