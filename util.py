""" Utility Functions """

import numpy as np
from PIL import Image
import cv2


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
        rotated_images.append(np.asarray(cur_image.rotate(90)))
        rotated_images.append(np.asarray(cur_image.rotate(180)))
        rotated_images.append(np.asarray(cur_image.rotate(270)))

    rotated_images = np.concatenate((images, np.array(rotated_images)), axis=0)

    if (unsqueeze):
        rotated_images = np.expand_dims(rotated_images, -1)

    return rotated_images


def padd_images(images, padding, padding_type=cv2.BORDER_REFLECT):
    """
    Padd padding on each side of the image
    :param images: np array of images
    :param padding: number of pixels to padd on each side
    :param padding_type: type of padding from cv2
    :return: np array of padded images
    """
    n = images.shape[0]
    return np.array([cv2.copyMakeBorder(images[i], padding, padding, padding, padding, padding_type) for i in range(n)])


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
        cur_image = np.zeros([img_height, img_width, patches.shape[-1]], dtype=np.uint8)
        for x in range(0, cur_image.shape[0], height):
            for y in range(0, cur_image.shape[1], width):
                cur_image[x:x + height, y:y + width] = patches[i + (x // height) * (img_height // height) + y // width]
                images.append(cur_image)
    return np.array(images)
