import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join


# convert grayscale image to strict black and white Image
def convert_to_bitmap(image, threshold):
    return cv.threshold(image, threshold, 255, cv.THRESH_BINARY)[1]


# thresholding
def thresholding(img):
    blur = cv.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return th3


# different modes of thresholding try out
def thresholding_test(img, plot_images):
    # global thresholding
    ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    if plot_images:
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
    return th3


# show Image, close on click? (t / f)
def show_image(name, im, c):
    cv.imshow(name, im)
    if c:
        cv.waitKey(0)


# label each contour with it's size )in pixels)
def show_contours(img):
    # convert to bitmap b/w
    thresh = thresholding(img)
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    total = 0
    im_x, im_y = img.shape
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        # create temp image (with mask set to white) to count non-zero pixels
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv.fillPoly(mask, [c], [255, 255, 255])
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        pixels = cv.countNonZero(mask)
        total += pixels
        y = y - 5 if y > im_y / 2 else y + h + 15
        cv.putText(thresh, '{}'.format(pixels), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (127, 127, 127), 1)

    show_image('thresh', thresh, True)


# get rid of all contours that are smaller than threshold (in pixels)
def tidy_contours(img, threshold):
    thresh = thresholding(img)
    # show_image('before', thresh, False)
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv.fillPoly(mask, [c], [255, 255, 255])
        # mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        pixels = cv.countNonZero(mask)
        if pixels < threshold:
            cv.fillPoly(thresh, pts=[c], color=(0, 0, 0))
    # show_image('after', thresh, True)
    return thresh

# simple probabilistic (take only subset of points) hough-transform
def hough_transform_probalistic():
    img = cv.imread('Results/Prediction_Images/test_10.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    show_image('houghLines', img, True)


# fully computed hough transform (over all pixels)
def hough_transform():
    img = cv.imread('Results/Prediction_Images/test_10.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    show_image('canny', edges, False)
    lines = cv.HoughLines(gray, 1, np.pi / 180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


# pipeline for post_processing im
def post_process(im):
    # post-processing Pipeline
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ## hough?
    # delete small blobs
    res = tidy_contours(gray, 200)
    return res


# entry point to post_process all pictures in dir_path and output them in output_path
def post_process_dir(dir_path, output_path):
    # get all png's in dir_path
    only_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(".png")]
    # for all png's: apply post-processing and save in output_path
    for file in only_files:
        image = cv.imread(dir_path + file)
        post_image = post_process(image)
        cv.imwrite(output_path + '/post_' + file, post_image)


# fit lines in all contours larger than threshold
def find_lines(threshold):
    # TODO process contours first: delete "arms"
    img = cv.imread('Results/Prediction_Images/test_12.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = thresholding(gray)
    contours, hier = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]

    # only fit contour if area over certain threshold, ie 1500
    for c in contours:
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv.fillPoly(mask, [c], [255, 255, 255])
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        pixels = cv.countNonZero(mask)
        if pixels > threshold:
            # then apply fitline() function
            [vx, vy, x, y] = cv.fitLine(c, cv.DIST_HUBER, 0, 0.01, 0.01)


            # Now find two extreme points on the line to draw line
            lefty = int((-x * vy / vx) + y)
            righty = int(((gray.shape[1] - x) * vy / vx) + y)

            # Finally draw the line in pink
            cv.line(img, (gray.shape[1] - 1, righty), (0, lefty), (127, 0, 255), 1)
    cv.imshow('img', img)
    # cv.imwrite('test.png', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    find_lines(1500)
    # post_process_dir("Results/Prediction_Images/", 'Results/Post_Processed')
