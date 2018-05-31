import os
import cv2 as cv

from matplotlib.cm import Greys_r as greyscale

img_dir = "./ims2"
import numpy as np
import PIL.Image as im
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


def showImage(iImage, iTitle=''):
    plt.figure()
    plt.imshow(iImage, cmap=cm.Greys_r)
    plt.suptitle(iTitle)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axes().set_aspect("equal", "datalim")
    plt.show()


file_list = os.listdir("./ims2")
for img_name in file_list:
    img_path = os.path.join(img_dir, img_name)
    print("Processing", img_path)
    pic = cv.imread(img_path)
    pic_gray = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)
    # pic_gray = pic_gray[60:330, 100:500]

    hist, edges = np.histogram(pic_gray, 64)
    max_idx = np.argmax(hist)
    intensity = edges[max_idx]
    pic_norm = np.asarray(pic_gray, dtype='float')
    pic_norm = (pic_norm / intensity) * 255
    pic_norm[pic_norm < np.iinfo('uint8').min] = np.iinfo('uint8').min
    pic_norm[pic_norm > np.iinfo('uint8').max] = np.iinfo('uint8').max

    pic_norm = pic_norm.astype('uint8')

    # showImage(pic_norm)

    canny_thr = 100
    edges = cv.Canny(pic_norm, canny_thr / 2, canny_thr)
    # imgBlurred = cv.GaussianBlur(pic_norm, (3, 3), 0)
    # 1, 60, param1=100, param2=10, minRadius=7, maxRadius=20
    circlesPlatform = cv.HoughCircles(pic_gray, cv.HOUGH_GRADIENT, 1, 1000, param1=100, param2=10, minRadius=180,
                                      maxRadius=230)
    circlesBall = cv.HoughCircles(pic_gray, cv.HOUGH_GRADIENT, 1, 60, param1=100, param2=10, minRadius=15, maxRadius=20)

    if circlesPlatform is None:
        continue

    circlesBall = np.uint16(np.around(circlesBall))
    for i in circlesBall[0, :]:
        cv.circle(pic_gray, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # print('Detected ball at ({},{})'.format(i[0], i[1]))
        cv.circle(pic_gray, (i[0], i[1]), 2, (0, 0, 255), 3)

    circlesPlatform = np.uint16(np.around(circlesPlatform))
    for i in circlesPlatform[0, :]:
        cv.circle(pic_gray, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # print('Detected ball at ({},{})'.format(i[0], i[1]))
        cv.circle(pic_gray, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv.imshow('Image', pic_gray)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join(img_dir, "detect_" + img_name), pic_gray)
