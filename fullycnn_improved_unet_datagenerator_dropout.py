# -*- coding: utf-8 -*-
"""fullyCNN_improved_unet_datagenerator_dropout.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jsEBqeQCnUV78PYE4b3Ndq-1ragFOTB0
"""

import numpy as np
import pandas as pd
import os, sys

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import concatenate
from keras import optimizers
from keras.layers import BatchNormalization, Activation
from keras.backend import binary_crossentropy
import keras
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from util.mask_to_submission import masks_to_submission
import util.util as util

## Install the following packages
import imageio
from PIL import Image
import cv2
import natsort
import logging

"""## Constants"""

# Name of the current model
MODEL_NAME = 'fullyCNN_datagenerator_improved_unet_dropout'
IMG_WIDTH = 608
IMG_HEIGHT = 608
EPOCHS = 200
STEPS_PER_EPOCH = 750
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.1
rnd_seed = 4
np.random.seed(rnd_seed)


logging.basicConfig(filename='Logs/messages.log', level=logging.INFO, format="%(asctime)s.%(msecs)03d[%(levelname)-8s]: ")
logging.info('Starting ' + MODEL_NAME)

"""## Load Images"""

logging.info('Loading training and test images')
training_image_dir = "training_images/images/"
training_label_dir = "training_images/groundtruth/"
test_image_dir = "test_images/normal/"

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
test_image = util.load_images(test_image_dir, files_test, "RGB")

print("TRAINING:")
print(training_image_original.shape)
print(training_label_original.shape)
print(training_image_extra.shape)
print(training_label_extra.shape)
print("TEST:")
print(test_image.shape)

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

# Convert image lists to numpy arrays for further processing
print(training_image_original.shape)
print(training_image_original.dtype)
print(training_label_original.shape)
print(training_image_extra.shape)
print(training_label_extra.shape)

# Create a validation set
training_image_original, validation_image, training_label_original, validation_label = train_test_split(
    training_image_original, training_label_original, test_size=VALIDATION_SPLIT, random_state=rnd_seed)

print(training_image_original.shape)
print(training_label_original.shape)
print(training_image_extra.shape)
print(training_label_extra.shape)
print(training_image_extra.shape)
print(validation_image.shape)
print(validation_label.shape)

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

print(training_image.shape)
print(training_label.shape)
print(validation_image.shape)
print(validation_label.shape)

# Rescale validation images/labels and test images because generator will do the same with training data
training_image = training_image.astype(np.float32)/255.0
training_label = training_label.astype(np.float32)/255.0
validation_image = validation_image.astype(np.float32)/255.0
validation_label = validation_label.astype(np.float32)/255.0
test_image = test_image.astype(np.float32)/255.0
logging.info('Finished Preprocessing!')

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

"""
## Loss Function and Accuracy Metric
- Accuracy: Intersection of prediction to label image over Union
- Loss :
    - Dice Coef Loss (https://arxiv.org/pdf/1606.04797v1.pdf)
    - Soft Dice Loss (Measure of interleaving of prediction image and label image)
    - Jaccard Distance

Source: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99
"""

from keras import backend as K

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def soft_dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def soft_dice_loss(y_true, y_pred):
    return 1-soft_dice_coef(y_true, y_pred)

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def jaccard_coef(y_true, y_pred, smooth = 1e-12):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def combined_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)

#def combined_loss(y_true, y_pred):
#    return dice_coef_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred)

"""## Model: Fully CNN built in Keras"""

def DilatedInceptionModule(inputs, numFilters = 32):
    #, dilation_rate = (2,2)
    x = Conv2D(numFilters, (3,3), padding='same', kernel_initializer = 'he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = Dropout(0.2) (x)

    x = Conv2D(numFilters, (3,3), padding='same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    return x

inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))

conv1 = DilatedInceptionModule(inputs, 16)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = DilatedInceptionModule(pool1, 32)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = DilatedInceptionModule(pool2, 64)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = DilatedInceptionModule(pool3, 128)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = DilatedInceptionModule(pool4, 256)
pool5 = MaxPooling2D(pool_size=(2,2)) (conv5)

conv6 = DilatedInceptionModule(pool5, 512)

up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (conv6)
concat7 = concatenate([up7, conv5])
conv7 = DilatedInceptionModule(concat7, 256)

up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv7)
concat8 = concatenate([up8, conv4])
conv8 = DilatedInceptionModule(concat8, 128)

up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv8)
concat9 = concatenate([up9, conv3])
conv9 = DilatedInceptionModule(concat9, 64)

up10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv9)
concat10 = concatenate([up10, conv2])
conv10 = DilatedInceptionModule(concat10, 32)

up11 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv10)
concat11 = concatenate([up11, conv1])
conv11 = DilatedInceptionModule(concat11, 16)

#crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
#conv9 = BatchNormalization()(crop9)
conv12 = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

model = Model(inputs=inputs, outputs=conv12)
model.summary()

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
early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, verbose=1)

"""## Model Training"""

opt = keras.optimizers.Adam(LEARNING_RATE)
#model = load_model("./Models/{}_model.h5".format(MODEL_NAME), custom_objects={'dice_coef_loss': dice_coef_loss, 'iou_coef': iou_coef})
opt = keras.optimizers.Nadam(lr=LEARNING_RATE)
logging.info('Compiling Model')
model.compile(
      optimizer=opt,
      loss=combined_loss,
      metrics=[iou_coef])

logging.info('Starting Training')

history = model.fit_generator(train_generator,
                              validation_data=(validation_image, validation_label),
                              steps_per_epoch=STEPS_PER_EPOCH,
                              epochs=EPOCHS,
                              callbacks=[checkpointer, csv_logger, lr_reducer, early_stopper])
logging.info('Finished Training')
"""
history = model.fit(training_image,
                    training_label,
                    validation_data=(validation_image, validation_label),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[checkpointer, csv_logger, lr_reducer, early_stopper]
                    )
"""
"""## Model Evaluation"""

# Kaggle scores on validation images (mean score per image and overall mean score)
logging.info('Evaluating Kaggle Score on Model')
model = load_model("./Models/{}_model.h5".format(MODEL_NAME), custom_objects={'combined_loss': dice_coef_loss, 'iou_coef': iou_coef})
predictions = model.predict(validation_image, batch_size=BATCH_SIZE, verbose=1)
scores = util.validate_kaggle_score(validation_label, predictions)
print(scores)
print(sum(scores)/len(scores))
logging.info('Score = ' + str(sum(scores)/len(scores)))

"""
## Create Submission File
Multiply image by 255 and convert to unit8 before storing s.t. it gets read out correctly by mask_to_submission!
"""

predictions = np.squeeze(predictions*255)
predictions = predictions.astype(np.uint8)
result_dir = './Results/Prediction_Images/{}/'.format(MODEL_NAME)
os.makedirs(result_dir, exist_ok=True)

[imageio.imwrite(result_dir + files_test[i], predictions[i],) for i in range(len(files_test))]
files_predictions = os.listdir(result_dir)
files_predictions = [result_dir + files_predictions[i] for i in range(len(files_predictions))]
masks_to_submission('./Results/Submissions/{}.csv'.format(MODEL_NAME), *files_predictions)
print('Submission ready')
