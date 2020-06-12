```python
# If running in Google Colab, mount drive
print('Check if running in Colab...')
try:
    from google.colab import drive
    print('Running in Colab!')
    drive.mount('/content/drive')
    %cd '/content/drive/My Drive/CIL-FS20'
except ImportError:
    print('Running locally!')

    #Check python version
    from platform import python_version
    print('Current python version: {}'.format(python_version()))

    # Check available GPUs
    import tensorflow as tf
    no_GPUs_available = len(tf.config.experimental.list_physical_devices('GPU'))
    print("Number of GPUs Available: {}".format(no_GPUs_available))
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
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import optimizers
from keras.layers import BatchNormalization
from tensorflow.keras.metrics import MeanIoU
from keras import backend as K
from keras.backend import binary_crossentropy
import keras
import random
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from datetime import datetime

from mask_to_submission import masks_to_submission
import util

## Install the following packages
import imageio
from PIL import Image
import cv2
import natsort
```

## Constants


```python
# Name of the current model
MODEL_NAME = 'fullyCNN_patches'

IMG_WIDTH = 400
IMG_HEIGHT = 400
PATCH_HEIGHT = 160
PATCH_WIDTH = 160
EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 8

# Set random seeds for reproducibility
rnd_seed = 4
np.random.seed(rnd_seed)
```

## Load Images


```python
training_images_dir = "training_images/images/"
training_labels_dir = "training_images/groundtruth/"
test_images_dir = "test_images/"

# Lad filenames
files_images = os.listdir(training_images_dir)
files_images = natsort.natsorted(files_images)
files_labels = os.listdir(training_labels_dir)
files_labels = natsort.natsorted(files_labels)
files_test = os.listdir(test_images_dir)
files_test = natsort.natsorted(files_test)
n = len(files_images)
n_test = len(files_test) 

# Load list of numpy arrays of training images and labels
print("Loading " + str(n) + " training images")
training_images_list = []
training_labels_list = []
for i in range(n):
    print("Loading training image {:04d}\r".format(i)),
    training_images_list.append(imageio.imread(training_images_dir + files_images[i]))
    training_labels_list.append(imageio.imread(training_labels_dir + files_labels[i]))

# Load list of numpy arrays of test images
print("Loading " + str(n_test) + " test images")
test_images_list = [imageio.imread(test_images_dir + files_test[i]) for i in range(n_test)]

# Convert lists to numpy arrays
training_images = np.array(training_images_list)
training_labels = np.expand_dims(np.array(training_labels_list), -1)
test_images = np.array(test_images_list)

print("TRAINING:")
print(training_images.shape)
print(training_labels.shape)
print("TEST:")
print(test_images.shape)
```


```python
# Make sure label masks only have values 1 or zero
#thresh_val = 0.5
#training_labels = (training_labels > thresh_val).astype(np.int64)
training_labels = training_labels/255
training_labels = training_labels.astype(np.float32)
print(training_labels.dtype)
#print(np.unique(training_labels, return_counts=True, axis=None))
```

## Preprocess Images
1. Randomize Training Images and take some of the training data as validation set
2. Split up Training Images into patches of size 160x160 (need to be divisible by 32!)
3. Split up Test Images into patches of size 152x152
4. Padd Test Images into size 160x160


```python
# Get a validation set
training_images, validation_images, training_labels, validation_labels = train_test_split(
    training_images, training_labels, test_size=0.1, random_state=rnd_seed)

# Image patchifying and padding
#validation_images = util.patchify(validation_images, 160, 160, 80)
#validation_labels = util.patchify(validation_labels, 160, 160, 80)
#training_images = util.patchify(training_images, 160, 160, 80)
#training_labels = util.patchify(training_labels, 160, 160, 80)
#test_images = util.patchify(test_images, 152, 152, 152)
#test_images = util.padd_images(test_images, 4)
test_images = util.patchify(test_images, 400, 400, 208)
print(test_images.dtype)
print(training_images.shape)
print(training_labels.shape)
print(validation_images.shape)
print(validation_labels.shape)
print(test_images.shape)
```

## Augment Training Data

Each training image can be rotated by 90 degrees and vertically an horizontally flipped. 
By doing so we increase our training data by a factor of 16.


```python
# flip training images horizontally, vertically and on both axes to increase training data *4
training_images = util.add_flipped_images(training_images)
training_images = util.add_rotated_images(training_images)
training_labels = util.add_flipped_images(training_labels)
training_labels = util.add_rotated_images(training_labels)

n = training_images.shape[0]
training_labels = np.squeeze(training_labels)

index = random.randint(0, n-1)
num_samples = 10
f = plt.figure(figsize = (15, 25))
for i in range(1, num_samples*2, 2):
  index = random.randint(0, n-1)

  f.add_subplot(num_samples, 2, i)
  plt.imshow(training_images[index])
  plt.title("Image")
  plt.axis('off')

  f.add_subplot(num_samples, 2, i+1)
  plt.imshow(training_labels[index])
  plt.title("Label")
  plt.axis('off')

plt.show()

training_labels = np.expand_dims(training_labels, -1)

print("Amount of training samples: " + str(n))
print(training_images.shape)
print(training_labels.shape)
```

## Loss Function and Accuracy Metric
- Accuracy: Intersection of prediction to label image over Union
- Loss :
    - Jaccard Distance (Current Best for patches)
    - Dice Coef Loss (https://arxiv.org/pdf/1606.04797v1.pdf)
    - Soft Dice Loss (Measure of interleaving of prediction image and label image)

Source: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99


```python
from keras import backend as K

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

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

def jaccard_coef(y_true, y_pred, smooth = 1e-12):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def combined_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)
```

## Model: Fully CNN built in Keras


```python
inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))

conv1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = keras.layers.advanced_activations.ELU()(conv1)
conv1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = keras.layers.advanced_activations.ELU()(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = keras.layers.advanced_activations.ELU()(conv2)
conv2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(conv2)
conv2 = BatchNormalization()(conv2)
conv2 = keras.layers.advanced_activations.ELU()(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = keras.layers.advanced_activations.ELU()(conv3)
conv3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = keras.layers.advanced_activations.ELU()(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(pool3)
conv4 = BatchNormalization()(conv4)
conv4 = keras.layers.advanced_activations.ELU()(conv4)
conv4 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = keras.layers.advanced_activations.ELU()(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(pool4)
conv5 = BatchNormalization()(conv5)
conv5 = keras.layers.advanced_activations.ELU()(conv5)
conv5 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(conv5)
conv5 = BatchNormalization()(conv5)
conv5 = keras.layers.advanced_activations.ELU()(conv5)

up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(up6)
conv6 = BatchNormalization()(conv6)
conv6 = keras.layers.advanced_activations.ELU()(conv6)
conv6 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(conv6)
conv6 = BatchNormalization()(conv6)
conv6 = keras.layers.advanced_activations.ELU()(conv6)

up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(up7)
conv7 = BatchNormalization()(conv7)
conv7 = keras.layers.advanced_activations.ELU()(conv7)
conv7 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(conv7)
conv7 = BatchNormalization()(conv7)
conv7 = keras.layers.advanced_activations.ELU()(conv7)

up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(up8)
conv8 = BatchNormalization()(conv8)
conv8 = keras.layers.advanced_activations.ELU()(conv8)
conv8 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(conv8)
conv8 = BatchNormalization()(conv8)
conv8 = keras.layers.advanced_activations.ELU()(conv8)

up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(up9)
conv9 = BatchNormalization()(conv9)
conv9 = keras.layers.advanced_activations.ELU()(conv9)
conv9 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(conv9)
#crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
#conv9 = BatchNormalization()(crop9)
conv9 = BatchNormalization() (conv9)
conv9 = keras.layers.advanced_activations.ELU()(conv9)
conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

model = Model(inputs=inputs, outputs=conv10)
model.summary()
```

## Callbacks for Observations


```python
#tbc=TensorBoardColab()
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
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
```

## Model Training


```python
#opt = keras.optimizers.adam(LEARNING_RATE)
opt = keras.optimizers.Nadam(lr=1e-4)
model.compile(
      optimizer=opt,
      loss=combined_loss,
      metrics=[iou_coef])
```


```python
# Labels are allready 1 or 0 now!
history = model.fit(training_images,
                    training_labels,
                    validation_data =(validation_images, validation_labels),
                    epochs=EPOCHS,
                    batch_size = BATCH_SIZE,
                    callbacks = [checkpointer, csv_logger, lr_reducer]
                    )
```


```python
# Show a training report
training_info = pd.read_csv('./Logs/{}_log.csv'.format(MODEL_NAME), header=0)

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
model = load_model("./Models/{}_model.h5".format(MODEL_NAME), custom_objects={'combined_loss': combined_loss, 'iou_coef': iou_coef})
#model.evaluate(test_images, test_label)
predictions = model.predict(test_image, batch_size=4, verbose=1)
```


```python
def unpatchify(patches, width, height, img_width, img_height, stride):
    """
    Reassembles array of patches to array of images
    :param patches: np array of patches
    :param width: patch width
    :param height: patch height
    :param stride: offset of one patch to next
    :return: np array of images
    """
    ''' TODO stride '''
    images = []
    if(patches.dtype == 'float32'):
        patches = patches / 4
    else:
        patches = patches // 4
    for i in range(0, patches.shape[0], (img_height // stride) * (img_width // stride)):
        cur_image = np.zeros([img_height, img_width, patches.shape[-1]], dtype=patches.dtype)
        for x in range(0, cur_image.shape[0]-height+1, stride):
            for y in range(0, cur_image.shape[1]-width+1, stride):
                cur_image[x:x + height, y:y + width] += patches[i + (x // stride) * (img_height // stride) + y // stride]
        images.append(cur_image)

    # if stride < img size we have to divide image intensities at overlap
    # TODO: this only works for current stride
    images = np.array(images)
    images[:, 0:stride, :] *= 2
    images[:, height:img_height, :] *= 2
    images[:, :, 0:stride] *= 2
    images[:, :, height:img_height] *= 2
    return images

test_images = unpatchify(test_images, 400, 400, 608, 608, 208)
predictions2 = unpatchify(predictions, 400, 400, 608, 608, 208)
```


```python
thresh_val = 0.25
predicton_threshold = (predictions > thresh_val).astype(np.uint8)

index = random.randint(0, len(predictions)-1)
num_samples = 10

f = plt.figure(figsize = (15, 25))
for i in range(1, num_samples*3, 3):
  index = random.randint(0, len(predictions2)-1)

  f.add_subplot(num_samples, 3, i)
  plt.imshow(test_images[index][:,:,0])
  plt.title("Image")
  plt.axis('off')

  f.add_subplot(num_samples, 3, i+1)
  plt.imshow(np.squeeze(predictions2[index][:,:,0]))
  plt.title("Prediction")
  plt.axis('off')

  f.add_subplot(num_samples, 3, i+2)
  plt.imshow(np.squeeze(predicton_threshold[index][:,:,0]))
  plt.title("Thresholded at {}".format(thresh_val))
  plt.axis('off')

plt.show()
```

## Create Submission File
Multiply image by 255 and convert to unit8 before storing s.t. it gets read out correctly by mask_to_submission!


```python
predictions = np.squeeze(predictions*255)
predictions = predictions.astype(np.uint8)
result_dir = './Results/Prediction_Images/{}/'.format(MODEL_NAME)
os.makedirs(result_dir, exist_ok=True)

#print(predictions.shape)
#[print(predictions[i].shape) for i in range(n_test)]

[imageio.imwrite(result_dir + files_test[i], predictions[i],) for i in range(n_test)]
files_predictions = os.listdir(result_dir)
files_predictions = [result_dir + files_predictions[i] for i in range(n_test)]
masks_to_submission('./Results/Submissions/{}.csv'.format(MODEL_NAME), *files_predictions)
print('Submission ready')
```


```python

```
