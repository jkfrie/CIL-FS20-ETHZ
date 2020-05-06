```python
# If running in Google Colab, pull repo and training data
try:
    from google.colab import drive
    import os

    # Clone repos if not done yet
    if 'CIL-FS20' not in os.getcwd():
      !git clone https://username:password@github.com/jasonkfriedman/CIL-FS20.git
      !git clone https://github.com/rmenta/CIL-FS20-Data.git

      # Run code inside repo
      %cd CIL-FS20

      # Use new training data
      !rm training_images/groundtruth/* training_images/images/*
      !mv ../CIL-FS20-Data/training_images/groundtruth/* training_images/groundtruth/
      !mv ../CIL-FS20-Data/training_images/images/* training_images/images/

    # Check if running on GPU
    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
      raise SystemError('GPU device not found')
except ImportError:
    print('Not running in Colab')
    pass
```


```python
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import optimizers
from keras.layers import BatchNormalization
from tensorflow.keras.metrics import MeanIoU
from keras import backend as K
import keras
import random
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from datetime import datetime

from mask_to_submission import masks_to_submission
```


```python
## Install the following packages
import imageio
from PIL import Image
import cv2
import natsort
```

## Constants


```python
IMG_WIDTH = 608
IMG_HEIGHT = 608
EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
rnd_seed = 4
```

## Load Images


```python
training_image_dir = "training_images/images/"
training_label_dir = "training_images/groundtruth/"
test_image_dir = "test_images/"

files_image = os.listdir(training_image_dir)
files_image = natsort.natsorted(files_image)
files_label = os.listdir(training_label_dir)
files_label = natsort.natsorted(files_label)
files_test = os.listdir(test_image_dir)
files_test = natsort.natsorted(files_test)
n = len(files_image)
n_test = len(files_test) 

# Load list of numpy arrays of training images and labels
print("Loading " + str(n) + " training images")
training_image_list = [imageio.imread(training_image_dir + files_image[i]) for i in range(n)]
training_label_list = [imageio.imread(training_label_dir + files_label[i]) for i in range(n)]

# Load list of numpy arrays of test images
print("Loading " + str(n_test) + " test images")
test_image_list = [imageio.imread(test_image_dir + files_test[i]) for i in range(n_test)]

print("TRAINING:")
print(training_image_list[0].shape)
print(training_label_list[0].shape)
print("TEST:")
print(test_image_list[0].shape)
```

## Padd Images
Training images have size 400x400 and test images have size 608x608. So we need to pad training images to same size, 
for that I use mirror padding for now.


```python
# Mirror padd all training images to get same size as test images
training_image_padded_list = [cv2.copyMakeBorder(training_image_list[i],104,104,104,104,cv2.BORDER_REFLECT) for i in range(n)]
training_label_padded_list = [cv2.copyMakeBorder(training_label_list[i],104,104,104,104,cv2.BORDER_REFLECT) for i in range(n)]

# Plot random Sample of images
index = random.randint(0, n-1)
num_samples = 5

f = plt.figure(figsize = (15, 25))
for i in range(1, num_samples*4, 4):
  index = random.randint(0, n-1)

  f.add_subplot(num_samples, 4, i)
  plt.imshow(training_image_list[index])
  plt.title("Original Image")
  plt.axis('off')

  f.add_subplot(num_samples, 4, i+1)
  plt.imshow(training_image_padded_list[index])
  plt.title("Padded Image")
  plt.axis('off')

  f.add_subplot(num_samples, 4, i+2)
  plt.imshow(np.squeeze(training_label_list[index]))
  plt.title("Original Label")
  plt.axis('off')

  f.add_subplot(num_samples, 4, i+3)
  plt.imshow(np.squeeze(training_label_padded_list[index]))
  plt.title("Padded Label")
  plt.axis('off')

plt.show()

# Convert image lists to numpy arrays for further processing
training_image = np.array(training_image_padded_list)
training_label = np.expand_dims(np.array(training_label_padded_list), -1)
test_image = np.array(test_image_list)
print(training_image.shape)
print(training_label.shape)
```

## Augment Training Data

Each training image can be rotated by 90 degrees and vertically an horizontally flipped. 
By doing so we increase our training data by a factor of 16.


```python
# flip training images horizontally, vertically and on both axes to increase training data *4
flipped_training_images = []
flipped_training_labels = []
for i in range(n):
    cur_image = Image.fromarray(training_image[i])
    cur_label = Image.fromarray(np.squeeze(training_label[i]))
    flipped_training_images.append(np.asarray(cur_image.transpose(Image.FLIP_LEFT_RIGHT)))
    flipped_training_labels.append(np.asarray(cur_label.transpose(Image.FLIP_LEFT_RIGHT)))
    cur_image = cur_image.transpose(Image.FLIP_TOP_BOTTOM)
    cur_label = cur_label.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_training_images.append(np.asarray(cur_image))
    flipped_training_labels.append(np.asarray(cur_label))
    flipped_training_images.append(np.asarray(cur_image.transpose(Image.FLIP_LEFT_RIGHT)))
    flipped_training_labels.append(np.asarray(cur_label.transpose(Image.FLIP_LEFT_RIGHT)))
    
training_image = np.concatenate((training_image, np.array(flipped_training_images)), axis=0)
training_label = np.concatenate((training_label, np.expand_dims(np.array(flipped_training_labels), -1)), axis=0)
n = training_image.shape[0]
print("Amount of training samples: " + str(n))
print(training_image.shape)
print(training_label.shape)

# Plot flipped images
f = plt.figure(figsize = (15, 25))

f.add_subplot(1, 3, 1)
plt.imshow(flipped_training_images[0])
plt.title("flipped vertical axis")
plt.axis('off')

f.add_subplot(1, 3, 2)
plt.imshow(flipped_training_images[1])
plt.title("flipped horizontal axis")
plt.axis('off')

f.add_subplot(1, 3, 3)
plt.imshow(flipped_training_images[2])
plt.title("flipped both axis")
plt.axis('off')

f.add_subplot(2, 3, 1)
plt.imshow(flipped_training_labels[0])
plt.title("flipped vertical axis")
plt.axis('off')

f.add_subplot(2, 3, 2)
plt.imshow(flipped_training_labels[1])
plt.title("flipped horizontal axis")
plt.axis('off')

f.add_subplot(2, 3, 3)
plt.imshow(flipped_training_labels[2])
plt.title("flipped both axis")
plt.axis('off')
plt.show()
```


```python
# rotate each training image by 90, 180 and 270 degrees to further increase training data *4
rotated_training_images = []
rotated_training_labels = []
for i in range(n):
    cur_image = Image.fromarray(training_image[i])
    cur_label = Image.fromarray(np.squeeze(training_label[i]))
    rotated_training_images.append(np.asarray(cur_image.rotate(90)))
    rotated_training_labels.append(np.asarray(cur_label.rotate(90)))
    rotated_training_images.append(np.asarray(cur_image.rotate(180)))
    rotated_training_labels.append(np.asarray(cur_label.rotate(180)))
    rotated_training_images.append(np.asarray(cur_image.rotate(270)))
    rotated_training_labels.append(np.asarray(cur_label.rotate(270)))
    
training_image = np.concatenate((training_image, np.array(rotated_training_images)), axis=0)
training_label = np.concatenate((training_label, np.expand_dims(np.array(rotated_training_labels), -1)), axis=0)
n = training_image.shape[0]
print("Amount of training samples: " + str(n))
print(training_image.shape)
print(training_label.shape)

# Plot rotated images
f = plt.figure(figsize = (15, 25))

f.add_subplot(1, 3, 1)
plt.imshow(rotated_training_images[0])
plt.title("+90 degrees")
plt.axis('off')

f.add_subplot(1, 3, 2)
plt.imshow(rotated_training_images[1])
plt.title("+180 degrees")
plt.axis('off')

f.add_subplot(1, 3, 3)
plt.imshow(rotated_training_images[2])
plt.title("+270 degrees")
plt.axis('off')

f.add_subplot(2, 3, 1)
plt.imshow(rotated_training_labels[0])
plt.title("+90 degrees")
plt.axis('off')

f.add_subplot(2, 3, 2)
plt.imshow(rotated_training_labels[1])
plt.title("+180 degrees")
plt.axis('off')

f.add_subplot(2, 3, 3)
plt.imshow(rotated_training_labels[2])
plt.title("+270 degrees")
plt.axis('off')
plt.show()
```

## Loss Function and Accuracy Metric
- Accuracy: Intersection of prediction to label image over Union
- Loss : Soft Dice Loss (Measure of interleaving of prediction image and label image)

Source: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99


```python
from keras import backend as K
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  
  return iou
```


```python
def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
```

## Model: Fully CNN built in Keras


```python
inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))
#s = Lambda(lambda x: x / 400) (inputs)

conv1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
conv1 = BatchNormalization() (conv1)
conv1 = Dropout(0.1) (conv1)
conv1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv1)
conv1 = BatchNormalization() (conv1)
pooling1 = MaxPooling2D((2, 2)) (conv1)

conv2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling1)
conv2 = BatchNormalization() (conv2)
conv2 = Dropout(0.1) (conv2)
conv2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv2)
conv2 = BatchNormalization() (conv2)
pooling2 = MaxPooling2D((2, 2)) (conv2)

conv3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling2)
conv3 = BatchNormalization() (conv3)
conv3 = Dropout(0.2) (conv3)
conv3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv3)
conv3 = BatchNormalization() (conv3)
pooling3 = MaxPooling2D((2, 2)) (conv3)

conv4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling3)
conv4 = BatchNormalization() (conv4)
conv4 = Dropout(0.2) (conv4)
conv4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv4)
conv4 = BatchNormalization() (conv4)
pooling4 = MaxPooling2D(pool_size=(2, 2)) (conv4)

conv5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling4)
conv5 = BatchNormalization() (conv5)
conv5 = Dropout(0.3) (conv5)
conv5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv5)
conv5 = BatchNormalization() (conv5)


upsample6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv5)
upsample6 = concatenate([upsample6, conv4])
conv6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample6)
conv6 = BatchNormalization() (conv6)
conv6 = Dropout(0.2) (conv6)
conv6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv6)
conv6 = BatchNormalization() (conv6)

upsample7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv6)
upsample7 = concatenate([upsample7, conv3])
conv7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample7)
conv7 = BatchNormalization() (conv7)
conv7 = Dropout(0.2) (conv7)
conv7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv7)
conv7 = BatchNormalization() (conv7)

upsample8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv7)
upsample8 = concatenate([upsample8, conv2])
conv8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample8)
conv8 = BatchNormalization() (conv8)
conv8 = Dropout(0.1) (conv8)
conv8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv8)
conv8 = BatchNormalization() (conv8)

upsample9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv8)
upsample9 = concatenate([upsample9, conv1], axis=3)
conv9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample9)
conv9 = BatchNormalization() (conv9)
conv9 = Dropout(0.1) (conv9)
conv9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv9)
conv9 = BatchNormalization() (conv9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)

model = Model(inputs=[inputs], outputs=[outputs])
model.summary()
```

## Callbacks for Observations


```python
#tbc=TensorBoardColab()
model_path = "./Models/fullyCNN_temp.h5"
checkpointer = ModelCheckpoint(model_path,
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)
csv_logger = CSVLogger("./Logs/fullyCNN_log.csv", separator=',', append=False)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=6,
                               verbose=1,
                               epsilon=1e-4)
```

## Model Training


```python
opt = keras.optimizers.adam(LEARNING_RATE)
model.compile(
      optimizer=opt,
      loss=soft_dice_loss,
      metrics=[iou_coef])
```


```python
# Divide labels by 255 s.t. we have labels between 0 an 1
history = model.fit(training_image,
                    training_label/255,
                    validation_split = 0.1,
                    epochs=EPOCHS,
                    batch_size = BATCH_SIZE,
                    callbacks = [checkpointer, csv_logger, lr_reducer]
                    )
```


```python
# Show a training report
training_info = pd.read_csv('./Logs/fullyCNN_log.csv', header=0)

acc1, = plt.plot(training_info['epoch'], training_info['iou_coef'])
acc2, = plt.plot(training_info['epoch'], training_info['val_iou_coef'])
plt.legend([acc1, acc2], ['Training IOU coef', 'Validation IOU coef'])
plt.xlabel('Epoch')
plt.ylim(0,1)
plt.grid(True)
plt.show()

loss1, = plt.plot(training_info['epoch'], training_info['loss'])
loss2, = plt.plot(training_info['epoch'], training_info['val_loss'])
plt.legend([acc1, acc2], ['Training Loss', 'Validation Loss'])                            
plt.xlabel('Epoch')
plt.ylim(0,1)
plt.grid(True)

plt.show()
```

## Model Evaluation


```python
model = load_model("./Models/fullyCNN_temp.h5", custom_objects={'soft_dice_loss': soft_dice_loss, 'iou_coef': iou_coef})
#model.evaluate(test_images, test_label)
predictions = model.predict(test_image, verbose=1)
```


```python
thresh_val = 0.5
predicton_threshold = (predictions > thresh_val).astype(np.uint8)

index = random.randint(0, len(predictions)-1)
num_samples = 10

f = plt.figure(figsize = (15, 25))
for i in range(1, num_samples*3, 3):
  index = random.randint(0, len(predictions)-1)

  f.add_subplot(num_samples, 3, i)
  plt.imshow(test_image[index][:,:,0])
  plt.title("Image")
  plt.axis('off')

  f.add_subplot(num_samples, 3, i+1)
  plt.imshow(np.squeeze(predictions[index][:,:,0]))
  plt.title("Prediction")
  plt.axis('off')

  f.add_subplot(num_samples, 3, i+2)
  plt.imshow(np.squeeze(predicton_threshold[index][:,:,0]))
  plt.title("Thresholded at {}".format(thresh_val))
  plt.axis('off')

plt.show()
```

## Create Submission File


```python
result_dir = './Results/Prediction_Images/'
[imageio.imwrite(result_dir + files_test[i], predictions[i]) for i in range(n_test)]
files_predictions = os.listdir(result_dir)
files_predictions = [result_dir + files_predictions[i] for i in range(n_test)]
masks_to_submission('./Results/Submissions/fullyCNN_baseline_19_April.csv', *files_predictions)
print('Submission ready')
```
