# import the necessary packages

import cv2
import numpy as np

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
image = cv2.imread("./demo/robust/capt20400.jpg")
pts = []

# mouse callback function
def draw_circle(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        pts.append(np.asarray((x, y)))


# Create a black image, a window and bind the function to window
img = image
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while len(pts) < 3:
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('e'):
        break

center = sum(pts) / 3
cv2.circle(img, (center[0].astype('int'), center[1].astype('int')), 4, (0, 0, 255), -1)

import math

radius = int(math.hypot(center[0] - pts[0][0], center[1] - pts[0][1]))

print(radius)

circlesPlatform = cv2.HoughCircles(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, 1000, param1=100,
                                   param2=10, minRadius=radius - 30,
                                   maxRadius=radius + 30)
if circlesPlatform is None:
    print('error')
circlesPlatform = np.uint16(np.around(circlesPlatform))
for i in circlesPlatform[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('img', img)

while True:
    k = cv2.waitKey(20) & 0xFF
    if k == ord('e'):
        break

print(center)
cv2.destroyAllWindows()
