import numpy as np
import cv2
import time
import sys
cap = cv2.VideoCapture(1)

idx = 20400
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)

    cv2.imwrite("./capt{}.jpg".format(idx), gray)

    idx += 1
    cv2.waitKey(1)


    input("enter to capture")
    print("Captured")
    sys.stdout.flush()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
