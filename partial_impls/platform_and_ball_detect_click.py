# import the necessary packages

import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not


# load the image, clone it, and setup the mouse callback function
image = cv2.imread("./partial_impls/capt20400.jpg")

import numpy as np

edge_pts = []


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global edge_pts
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        pts.append(np.asarray((x, y)))


# Create a black image, a window and bind the function to window
img = image
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while len(edge_pts) < 3:
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('e'):
        break

center = sum(edge_pts) / 3
cv2.circle(img, (center[0].astype('int'), center[1].astype('int')), 4, (0, 0, 255), -1)

import math

radius = int(math.hypot(center[0] - edge_pts[0][0], center[1] - edge_pts[0][1]))

print(radius)
print(edge_pts)

circlesPlatform = cv2.HoughCircles(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, 1000, param1=100,
                                   param2=10, minRadius=radius - 30,
                                   maxRadius=radius + 30)
if circlesPlatform is None:
    print('error')
circlesPlatform = np.uint16(np.around(circlesPlatform))
for i in circlesPlatform[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

balls = cv2.HoughCircles(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, 1000, param1=100,
                         param2=10, minRadius=int((radius - 30) / 13),
                         maxRadius=int((radius + 30) / 13))
if balls is None:
    print('error')
else:
    balls = np.uint16(np.around(balls))
    for i in balls[0, :]:
        j = np.array([109,250,17])
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1, 8, 0)

        # Apply mask (using bitwise & operator)
        result_array = image & mask
        print(i[2])

        # Crop/center result (assuming max_loc is of the form (x, y))
        result_array = result_array[i[1] - (i[2]/2):i[1] + (i[2]/2),
                       i[0] - (i[2]/2):i[0] + (i[2]/2), :]

        result_array = np.array(cv2.cvtColor(result_array, cv2.COLOR_BGR2GRAY))
        result_array = result_array[result_array > 0].flatten()
        print(np.mean(result_array))
        print(result_array)

cv2.imshow('img', img)

while True:
    k = cv2.waitKey(20) & 0xFF
    if k == ord('e'):
        break

print(center)
cv2.destroyAllWindows()
