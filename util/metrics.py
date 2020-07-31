""" Accuracy and Loss functions used for the training process """

from keras import backend as K
from keras.backend import binary_crossentropy

def iou_coef(y_true, y_pred, smooth=1):
    """
    Intersectin over Union
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3])+K.sum(y_pred, [1, 2, 3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coef_loss(y_true, y_pred, smooth=1):
    """
    Dice Coefficient loss function
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return 1 - (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

def jaccard_distance(y_true, y_pred, smooth = 1e-12):
    """
    Jaccard Coefficient
    """
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    jac_sum = K.sum(y_true + y_pred, axis=[0, 1, 2])
    jac = (intersection + smooth) / (jac_sum - intersection + smooth)
    return K.mean(jac)

def combined_loss(y_true, y_pred):
    """
    Combined loss function from jaccard coefficient and binary crossentropy
    """
    return -K.log(jaccard_distance(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)