""" Utility Functions used throughout the pipeline"""

import numpy as np
from PIL import Image
import cv2
import os
import imageio
from scipy import ndimage
from sklearn.metrics import f1_score
from util.mask_to_submission import nparray_masks_to_patch_labels
from util.mask_to_submission import masks_to_submission
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization, Activation

def load_images(path, filenames, pilmode):
    """
    Load all images with respective paths into an np.array
    :param path: path dir holding the images
    :param filenames: list of names of the images
    :param pilmode: pilmode from imageio.imread ("RGB" or "L")
    :return: np array of loaded images
    """
    images = []
    n = len(filenames)
    print("Loading " + str(n) + " images")
    images = [imageio.imread(path + filenames[i], pilmode=pilmode) for i in range(n)]
    return np.array(images)

def add_flipped_images(images):
    """
    Adds vertically, horizontally and bothways flipped images
    :param images: np array of images
    :return: np array of original + flipped images
    """
    flipped_images = []
    unsqueeze = False

    if (images.shape[-1] == 1):
        unsqueeze = True
        images = np.squeeze(images)

    for i in range(images.shape[0]):
        cur_image = Image.fromarray(images[i])
        flipped_images.append(np.asarray(cur_image.transpose(Image.FLIP_LEFT_RIGHT)))
        cur_image = cur_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_images.append(np.asarray(cur_image))
        flipped_images.append(np.asarray(cur_image.transpose(Image.FLIP_LEFT_RIGHT)))

    flipped_images = np.concatenate((images, np.array(flipped_images)), axis=0)

    if (unsqueeze):
        flipped_images = np.expand_dims(flipped_images, -1)

    return flipped_images


def add_rotated_images(images):
    """
    Adds all rotations by 90 degrees of the images
    :param images: np array of images
    :return: np array of original + rotated images
    """
    rotated_images = []
    unsqueeze = False

    if (images.shape[-1] == 1):
        unsqueeze = True
        images = np.squeeze(images)

    for i in range(images.shape[0]):
        cur_image = Image.fromarray(images[i])
        rotated_images.append(ndimage.rotate(cur_image, 45, reshape=False, mode='reflect'))
        rotated_images.append(np.asarray(cur_image.rotate(90)))
        rotated_images.append(ndimage.rotate(cur_image, 135, reshape=False, mode='reflect'))
        rotated_images.append(np.asarray(cur_image.rotate(180)))
        rotated_images.append(ndimage.rotate(cur_image, 225, reshape=False, mode='reflect'))
        rotated_images.append(np.asarray(cur_image.rotate(270)))
        rotated_images.append(ndimage.rotate(cur_image, 315, reshape=False, mode='reflect'))

    rotated_images = np.concatenate((images, np.array(rotated_images)), axis=0)

    if (unsqueeze):
        rotated_images = np.expand_dims(rotated_images, -1)

    return rotated_images


def padd_images(images, width, height, padding_type=cv2.BORDER_REFLECT):
    """
    Padd padding on each side of the image
    :param images: np array of images
    :param width: Up to which width to pad
    :param height: Up to which height to pad
    :param padding_type: type of padding from cv2
    :return: np array of padded images
    """
    images = images.tolist()
    for i in range(len(images)):
      cur_image = np.array(images[i])
      imheight = cur_image.shape[0]
      imwidth = cur_image.shape[1]
      pad_y = int((height - imheight) / 2)
      pad_x = int((width - imwidth) / 2)
      images[i] = cv2.copyMakeBorder(cur_image, pad_y, pad_y, pad_x, pad_x, padding_type)
    return np.array(images)


def crop_images(images, padding):
    """
    Crop padding on each side of the image
    :param images: np array of images
    :param padding: amount of pixels to crop on each side
    :return: np array of cropped images
    """
    height = images.shape[1]
    width = images.shape[2]
    return images[:, padding:height-padding, padding:width-padding]


def patchify(images, width, height, stride):
    """
    Splits up images of np.array 'images' into patches with 'width' and 'height',
    where stride determines the offset of next patch in image
    :param images: np array of images
    :param width: patch width
    :param height: patch height
    :param stride: offset of one patch to next
    :return: np array of patches
    """
    patchified_images = []
    x_lim = images.shape[1] - height + stride
    y_lim = images.shape[2] - width + stride
    for i in range(images.shape[0]):
        cur_image = images[i]
        patches = [cur_image[x:x + height, y:y + width] for x in range(0, x_lim, stride) for y in
                   range(0, y_lim, stride)]
        patchified_images = patchified_images + patches
    return np.array(patchified_images)


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
    for i in range(0, patches.shape[0], img_height // height * img_width // width):
        cur_image = np.zeros([img_height, img_width, patches.shape[-1]], dtype=np.float64)
        for x in range(0, cur_image.shape[0], height):
            for y in range(0, cur_image.shape[1], width):
                cur_image[x:x + height, y:y + width] = patches[i + (x // height) * (img_height // height) + y // width]
                images.append(cur_image)
    return np.array(images)

def validate_kaggle_score(y_true, y_pred):
    """
    Validate according to kaggle metric
    :param y_true: true masks
    :param y_pred: predicted masks
    :return: score
    """
    y_true = nparray_masks_to_patch_labels(y_true)
    y_pred = nparray_masks_to_patch_labels(y_pred)
    result = []
    for k in range(y_true.shape[0]):
        result.append(f1_score(y_true[k], y_pred[k], zero_division=1))
    return result

def create_submission(predictions, result_dir, filename, files_test):
    """
    Create a submission file
    :param predictions: predicted masks
    :param result_dir: directory to store results
    :param filename: name and path of the submission file
    :param files_test: list of test image names
    """
    n_test = len(files_test)
    # predictions = np.squeeze(predictions * 255).astype(np.uint8)
    # os.makedirs(result_dir, exist_ok=True)
    # [imageio.imwrite(result_dir + files_test[i], predictions[i], ) for i in range(n_test)]
    files_predictions = os.listdir(result_dir)
    files_predictions = [result_dir + files_predictions[i] for i in range(n_test)]
    masks_to_submission(filename, *files_predictions)
    print('Submission ready')

def convolutional_block(inputs, numFilters = 32):
    """
    A Convolutional Block for the U-Net module
    :param inputs: Input
    :param numFilters: nr of output channels
    :return: Output
    """
    x = Conv2D(numFilters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = Dropout(0.2)(x)

    x = Conv2D(numFilters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    return x
