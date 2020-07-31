import numpy as np
import imageio
from PIL import Image
import cv2 as cv
import natsort
import os
import matplotlib.pyplot as plt


# Constants
ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]
ROT_MARGIN = 89
h, w = 608, 608


# Rotate Image by angle degrees
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


# helper function to plot images with subplots
def add_to_subplot(img, axs, x, y, title):
    axs[x, y].imshow(img)
    axs[x, y].yaxis.set_visible(False)
    axs[x, y].xaxis.set_visible(False)
    axs[x, y].set_title(title)


# print all eight prediction images
def print_all_8(temp_im, name):
    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    for i in range(8):
        add_to_subplot(temp_im[i], axs, i % 3, int(i / 3), name + ' ' + str(ROTATIONS[i]))


# separate the rotated Images into 45 / 90 degree rotations
def split_90_45(ar):
    angle_90s = np.zeros((4, h, w), np.float)
    angle_45s = np.zeros((4, h, w), np.float)
    c45, c90 = 0, 0
    for x in range(8):
        if ROTATIONS[x] % 90 != 0:
            angle_45s[c45] = ar[x]
            c45 = c45 + 1
        else:
            angle_90s[c90] = ar[x]
            c90 = c90 + 1
    return angle_90s, angle_45s


# split the image into 4 quadrants, tl = top-left / br = bottom-right
def split_into_quadrants(ar):
    x_len = len(ar)
    y_len = len(ar[0])
    tl = ar[0:int(x_len / 2), 0:int(y_len / 2)]
    tr = ar[0:int(x_len / 2), int(y_len / 2): y_len]
    bl = ar[int(x_len / 2):x_len, 0:int(y_len / 2)]
    br = ar[int(x_len / 2):x_len, int(y_len / 2): y_len]
    return tl, tr, bl, br


# create one Image out of 4 quadrants, tl = top-left / br = bottom-right
def combine_quadrants(tl, tr, bl, br):
    top = np.column_stack((tl, tr))
    bot = np.column_stack((bl, br))
    tog = np.row_stack((top, bot))
    return tog


# combine the four quadrants of the 45 / 90 degree rotated predictions
def comb_45_90(tl90, tr90, bl90, br90, tl45, tr45, bl45, br45):
    tl = combine_it(tl90, tl45, 1)
    tr = combine_it(tr90, tr45, 2)
    bl = combine_it(bl90, bl45, 3)
    br = combine_it(br90, br45, 4)
    return combine_quadrants(tl, tr, bl, br)


# helper method for comb_45_90()
def combine_it(q90, q45, quadrant):
    x_len = len(q90[0])
    y_len = len(q90)
    res = np.zeros((x_len, y_len), np.float)
    if quadrant == 1:
        # top left quadrant
        for x in range(x_len):
            for y in range(y_len):
                if x + y < x_len:
                    res[x][y] = q90[x][y]
                else:
                    res[x][y] = (q90[x][y] + q45[x][y]) / 2
    elif quadrant == 2:
        # top right quadrant
        for x in range(x_len):
            for y in range(y_len):
                if x < y + 1:
                    res[x][y] = q90[x][y]
                else:
                    res[x][y] = (q90[x][y] + q45[x][y]) / 2
    elif quadrant == 3:
        # bottom left quadrant
        for x in range(x_len):
            for y in range(y_len):
                if x + 1 > y:
                    res[x][y] = q90[x][y]
                else:
                    res[x][y] = (q90[x][y] + q45[x][y]) / 2
    elif quadrant == 4:
        # bottom right quadrant
        for x in range(x_len):
            for y in range(y_len):
                if x + y >= x_len - 1:
                    res[x][y] = q90[x][y]
                else:
                    res[x][y] = (q90[x][y] + q45[x][y]) / 2
    return res


# fuse the 8 predictions of the same Iamge together in different ways
def eight_to_one(ar, mode):
    res = ar[0]
    a90, a45 = split_90_45(ar)
    if mode == 'MAX':
        max_ar = np.array(np.max(ar, axis=0), np.float)
        res = thresholding(np.array((max_ar), dtype=np.uint8))
    elif mode == 'MEAN':
        mean = np.array(np.mean(ar, axis=0), np.float)
        res = np.array((mean), dtype=np.uint8)
    elif mode == 'SQUARE_MEAN':
        square = np.array(np.square(ar), np.float)
        mean = np.array(np.mean(square, axis=0), np.float)
        res = thresholding(np.array((mean), dtype=np.uint8))
    elif mode == 'only90_mean':
        mean = np.array(np.mean(a90, axis=0), np.float)
        res = thresholding_otsu(np.array((mean), dtype=np.uint8))
    elif mode == 'only90_max':
        max_ar = np.array(np.max(a90, axis=0), np.float)
        res = thresholding(np.array((max_ar), dtype=np.uint8))
    elif mode == 'only90_max_no_th':
        max_ar = np.array(np.max(a90, axis=0), np.float)
        res = np.array(max_ar, dtype=np.uint8)
    elif mode == 'only90_max_tidy_contours':
        max_ar = np.array(np.max(a90, axis=0), np.float)
        res = thresholding(np.array((max_ar), dtype=np.uint8))
        res = tidy_contours(res, 500)
    elif mode == 'median':
        medi = np.array(np.median(ar, axis=0), np.float)
        res = thresholding(np.array((medi), dtype=np.uint8))
    elif mode == 'median_90s':
        medi = np.array(np.median(a90, axis=0), np.float)
        res = thresholding(np.array((medi), dtype=np.uint8))
    elif mode == 'max_low_threshold':
        max_ar = np.array(np.max(ar, axis=0), np.uint8)
        ret3, res = cv.threshold(max_ar, 0, 255, cv.THRESH_BINARY)
    elif mode == 'aggregate_and_count_10p':
        f = lambda x: sum(i > 0.1 for i in x)
        max_ar = np.array(np.apply_along_axis(f, 0, ar), np.float)
        res = thresholding(np.array((max_ar), dtype=np.uint8))
    elif mode == 'aggregate_and_count_50p':
        f = lambda x: sum(i > 0.5 for i in x)
        max_ar = np.array(np.apply_along_axis(f, 0, ar), np.float)
        res = thresholding(np.array((max_ar), dtype=np.uint8))
    elif mode == 'aggregate_and_count_tidy_contours':
        f = lambda x: sum(i > 0.1 for i in x)
        max_ar = np.array(np.apply_along_axis(f, 0, ar), np.float)
        res = thresholding(np.array((max_ar), dtype=np.uint8))
        res = tidy_contours(res, 400)
    elif mode == 'aggregate_and_count_45_90_dif_65':
        # difference in 45 / 90 degree pics how to handle to prevent more info in middle
        f = lambda x: sum(i > 0.65 for i in x)
        sum_90 = np.array(np.apply_along_axis(f, 0, a90), np.float)
        tl90, tr90, bl90, br90 = split_into_quadrants(sum_90)
        sum_45 = np.array(np.apply_along_axis(f, 0, a45), np.float)
        tl45, tr45, bl45, br45 = split_into_quadrants(sum_45)
        max_ar = comb_45_90(tl90, tr90, bl90, br90, tl45, tr45, bl45, br45)
        res = thresholding(np.array((max_ar), dtype=np.uint8))
        res = tidy_contours(res, 400)
    elif mode == 'mean_45_90_dif':
        mean_90 = np.array(np.mean(a90, axis=0), np.float)
        tl90, tr90, bl90, br90 = split_into_quadrants(mean_90)
        mean_45 = np.array(np.mean(a45, axis=0), np.float)
        tl45, tr45, bl45, br45 = split_into_quadrants(mean_45)
        max_ar = comb_45_90(tl90, tr90, bl90, br90, tl45, tr45, bl45, br45)
        res = thresholding(np.array((max_ar), dtype=np.uint8))
    elif mode == 'mean_45_90_dif_sq':
        mean_90 = np.array(np.mean(a90, axis=0), np.float)
        tl90, tr90, bl90, br90 = split_into_quadrants(mean_90)
        mean_45 = np.array(np.mean(a45, axis=0), np.float)
        tl45, tr45, bl45, br45 = split_into_quadrants(mean_45)
        max_ar = comb_45_90(tl90, tr90, bl90, br90, tl45, tr45, bl45, br45)
        max_ar = np.square(max_ar)
        res = thresholding(np.array((max_ar), dtype=np.uint8))
    return res


# convert grayscale image to strict black and white Image
def convert_to_bitmap(image, threshold):
    return cv.threshold(image, threshold, 255, cv.THRESH_BINARY)[1]


# thresholding
def thresholding(img):
    blur = cv.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return th3


# otsu - thresholding
def thresholding_otsu(img):
    ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return th2


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


# Load all test images, rotate them with ROTATIONS and store them in a list
def prepare_test_images(n_test, test_image_dir, files_test, rotated_test_image_dir):
    print('Rotating {} test images'.format(n_test))
    test_image_list = []
    margin = ROT_MARGIN
    for i in range(n_test):
        orig = Image.open(test_image_dir + files_test[i])
        # rotate each image 0, 45, 90, 135, 180, 225, 270, 315 degrees
        for r in ROTATIONS:
            width, height = orig.size
            im2 = orig.rotate(r)
            if r % 90 != 0:
                # crop
                w1, h1 = im2.size
                im2 = im2.crop((margin, margin, w1 - margin, h1 - margin)).resize((width, height),
                                                                                  resample=Image.BICUBIC)
            name = files_test[i].split('.png')
            im2.save(rotated_test_image_dir + name[0] + '_' + str(r) + '.png')
        # print('\tRotating ' + files_test[i])

    new_test_im_dir = os.listdir(rotated_test_image_dir)
    new_test_im_dir = natsort.natsorted(new_test_im_dir)
    test_image_list = [imageio.imread(rotated_test_image_dir + file) for file in new_test_im_dir]
    return test_image_list


# combine all 8 predicted Images with MODE and save to RESULT_DIR
def combine_and_save(mode, result_dir, predictions, files_test):
    # since test_images contains 8 Images per prediction test_image -> average over these 8 images
    count = 0
    dif_rot = len(ROTATIONS)
    temp_im = np.zeros((8, h, w), np.float)  # hold all 8 pictures, then flatten along axis=0
    new_predictions = np.zeros((94, h, w), dtype=np.uint8)
    # start = random.randint(0,94) * 8
    start = -1
    for ti in predictions:
        # read in image
        img = cv.resize(ti, (608, 608))
        ind = count % len(ROTATIONS)

        if ROTATIONS[ind] % 90 != 0:
            # 45 degree rotation
            # first downsample
            img = cv.resize(img, (430, 430), interpolation=cv.INTER_CUBIC)
            # add margin
            img = cv.copyMakeBorder(img, 89, 89, 89, 89, cv.BORDER_CONSTANT, value=[0, 0, 0])
            # reverse rotation
            img = rotate_image(img, -ROTATIONS[ind])
            # resample to w,h

        else:
            # its a 90 degree rotation
            img = rotate_image(img, -ROTATIONS[ind])

        # last pic is never saved
        if count % dif_rot == 0 and count != 0:
            if count == start:
                print_all_8(temp_im, files_test[int(count / len(ROTATIONS)) - 1])
            # previous was the last pic
            final_pred = eight_to_one(temp_im, mode)
            new_predictions[int(count / 8)] = final_pred
            out = Image.fromarray(final_pred, mode='L')
            imageio.imwrite(result_dir + files_test[int(count / len(ROTATIONS)) - 1], out)
            # reset, new pic
            temp_im = np.zeros((8, h, w), np.float)
        # build array with all 8 pics
        temp_im[count % 8] = img
        count = count + 1

    # last pic save
    final_pred = eight_to_one(temp_im, mode)
    new_predictions[count % dif_rot] = final_pred
    out = Image.fromarray(final_pred, mode='L')
    imageio.imwrite(result_dir + files_test[int(count / len(ROTATIONS)) - 1], out)


if __name__ == '__main__':
    print('hi')
