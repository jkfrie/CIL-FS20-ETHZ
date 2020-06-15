import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

prediction_images_path = "Results/Prediction_Images/"


# convert grayscale image to strict black and white Image
def convert_to_bitmap(image, threshold):
    return cv.threshold(image, threshold, 255, cv.THRESH_BINARY)[1]


# different modes of thresholding try out
def thresholding(img):
    # global thresholding
    ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()

# show Image, close on click
def show_image(im):
    cv.imshow(im)
    cv.waitKey(0)


def test():
    thresh = convert_to_bitmap(image, 127)
    cnts = cv.findContours(thresh)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    total = 0

    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv.fillPoly(mask, [c], [255, 255, 255])
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        pixels = cv.countNonZero(mask)
        total += pixels
        cv.putText(image, '{}'.format(pixels), (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    print(total)
    cv.imshow('thresh', thresh)
    cv.imshow('image', image)
    cv.waitKey(0)


if __name__ == '__main__':
    image = cv.imread('Results/Prediction_Images/test_7.png')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresholding(gray)
