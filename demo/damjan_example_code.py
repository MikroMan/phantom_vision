# import the necessary packages
import socket
import struct
import sys
import time

import cv2
import cv2 as cv
import numpy as np

# client_ip = "192.168.65.94"
client_ip = "192.168.65.94"
port = 26000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP


def send_vals(x, y, d):
    vals = (x, y, d)
    packer = struct.Struct('f f f')
    bin_data = packer.pack(*vals)
    sock.sendto(bin_data, (client_ip, port))


cap = cv2.VideoCapture(0)
# for i in range(50):
ret, image = cap.read()
blur = cv2.GaussianBlur(image, (9, 9), 0)
hsv = cv2.cvtColor(blur, cv.COLOR_BGR2HSV)
h = hsv[:, :, 0]
s = hsv[:, :, 1]
v = hsv[:, :, 2]

# rvlib.showImage(blur)

# pts=plt.ginput(n=4, timeout=-1)
# np.save('coardUV.npy', pts)
# pts=np.load('coardUV.npy')
'''Blue=h[int(pts[0][1]), int(pts[0][0])]/255.0*360
Green=h[int(pts[1][1]), int(pts[1][0])]/255.0*360
Red=h[int(pts[2][1]), int(pts[2][0])]/255.0*360
BallH=h[int(pts[3][1]), int(pts[3][0])]/255.0*360
BallV=v[int(pts[3][1]), int(pts[3][0])]/255.0*100'''
Blue = 144.0
Green = 115.8
Red = 248.5
BallH = 127.0
BallV = 34.1

# while(True):
BallPos = np.array([0, 0])
Center = np.array([0, 0])
RedMarker = np.array([0, 0])
BlueMarker = np.array([0, 0])
GreenMarker = np.array([0, 0])
while (1):
    # Capture frame-by-frame
    # imageC = cv.imread('Slika1.jpg')

    # output = image.copy()
    # output = np.array(image[:,:,2])
    blur = cv2.GaussianBlur(image, (9, 9), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    # h = np.array(h,dtype = 'uint8')
    s = hsv[:, :, 1]
    s = np.array(s, dtype='uint8')
    v = hsv[:, :, 2]
    v = np.array(v, dtype='uint8')

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=108, param2=20, minRadius=5, maxRadius=30)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        blue = 0
        blueMarkers = []
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            if (x < 1920):
                hue = h[y][x] / 255 * 360
                val = v[y][x] / 255.0 * 100
                sat = s[y][x] / 255.0 * 100

                if ((((Blue - 10) < hue) and (hue < (Blue + 10))) and val > 40):  # Modra barva
                    cv2.circle(image, (x, y), r, (255, 0, 0), 4)
                    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    BlueMarker = np.array([x, y])
                    # print('Modra: '+str(hue))
                elif ((((Red - 10) < hue) and (hue < (Red + 10))) and sat > 50):  # Rdeča barva
                    cv2.circle(image, (x, y), r, (0, 0, 255), 4)
                    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    RedMarker = np.array([x, y])
                    # print('Rdeča: '+str(hue))
                elif ((((Green - 10) < hue) and (hue < (Green + 10))) and sat > 50):  # Zelena barva
                    cv2.circle(image, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    GreenMarker = np.array([x, y])
                    # print('Zelena: '+str(hue))
                elif ((val < (BallV + 20)) and ((BallV - 20) < val)):  # or (100 < col < 170): Siva barva
                    cv2.circle(image, (x, y), r, (0, 0, 0), 4)
                    rref = r
                    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    BallPos = np.array([x, y])
                    print(r)
                    '''print(h[y][x])
                    print(hue)
                    print(sat)
                    print(val)
                    print(' ')

                else:
                    print(h[y][x]*360/255)
                    print(s[y][x]*100/255)
                    print(v[y][x]*100/255)
                    print('  ')'''

        # Določi referenčni center
        Center = np.array([int((GreenMarker[0] + BlueMarker[0] + RedMarker[0]) / 3),
                           int((GreenMarker[1] + BlueMarker[1] + RedMarker[1]) / 3)])
        # Center[1]=np.array(int((GreenMarker[1]+BlueMarker[1]+RedMarker[1])/3))
        cv2.circle(image, (Center[0], Center[1]), rref, (10, 100, 100), 4)
        cv2.rectangle(image, (Center[0] - 5, Center[1] - 5), (Center[0] + 5, Center[1] + 5), (0, 128, 255), -1)
        cv2.line(image, (Center[0], Center[1]), (BallPos[0], BallPos[1]), (255, 0, 0), 2)

        # Določi x in y osi ter enotski matriki
        Xos = np.array([(-BlueMarker[0] + GreenMarker[0]), ((-BlueMarker[1] + GreenMarker[1]))])
        XosE = Xos / np.linalg.norm(Xos)

        Yos = np.array([Xos[1], -Xos[0]])
        YosE = Yos / np.linalg.norm(Yos)

        # Pretvorba iz px v mm
        kx = 245 / np.linalg.norm(Xos)

        ptmp = np.array([int((BlueMarker[0] + GreenMarker[0]) / 2), int((BlueMarker[1] + GreenMarker[1]) / 2)])
        cv2.rectangle(image, (ptmp[0] - 5, ptmp[1] - 5), (ptmp[0] + 5, ptmp[1] + 5), (0, 128, 255), -1)
        ytmp = np.array([(-ptmp[0] + RedMarker[0]), ((-ptmp[1] + RedMarker[1]))])
        ky = 200 / np.linalg.norm(ytmp)

        # Določanje rotacijske matrike met ks slike in ks robota
        R = np.array([[XosE[0], XosE[1]], [YosE[0], YosE[1]]])
        Vy = np.matmul(R, YosE)
        Vx = np.matmul(R, XosE)

        # Določanje vektorja odmika, smer in absolutna vrednost
        Rold = (-BallPos + Center)
        RdifE = (-BallPos + Center) / np.linalg.norm((-BallPos + Center))
        RdiffG = np.matmul(R, RdifE)
        Absdiff = np.linalg.norm((-BallPos + Center)) * kx

        if (len(circles) == 4):
            break

    # show the output image
    cv2.imshow("output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#########
while 1:
    T = time.time()
    # Capture frame-by-frame
    ret, image = cap.read()
    # image = cv2.imread('Krogi.jpg')

    # output = image.copy()
    # output = np.array(image[:,:,2])
    blur = cv2.GaussianBlur(image, (9, 9), 0)
    hsv = cv2.cvtColor(blur, cv.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    # h = np.array(h,dtype = 'uint8')
    s = hsv[:, :, 1]
    s = np.array(s, dtype='uint8')
    v = hsv[:, :, 2]
    v = np.array(v, dtype='uint8')

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=108, param2=20, minRadius=5, maxRadius=20)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        blue = 0
        blueMarkers = []
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            if (y < 480):
                hue = h[y][x] / 255.0 * 360
                val = v[y][x] / 255.0 * 100
                sat = s[y][x] / 255.0 * 100

                if ((((Blue - 10) < hue) and (hue < (Blue + 10))) and val > 40):  # Modra barva
                    cv2.circle(image, (x, y), r, (255, 0, 0), 4)
                    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    BlueMarker = np.array([x, y])
                    # print('Modra: '+str(hue))
                elif ((((Red - 10) < hue) and (hue < (Red + 10))) and sat > 50):  # Rdeča barva
                    cv2.circle(image, (x, y), r, (0, 0, 255), 4)
                    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    RedMarker = np.array([x, y])
                    # print('Rdeča: '+str(hue))
                elif ((((Green - 10) < hue) and (hue < (Green + 10))) and sat > 50):  # Zelena barva
                    cv2.circle(image, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    GreenMarker = np.array([x, y])
                    # print('Zelena: '+str(hue))
                elif ((val < (BallV + 20)) and ((BallV - 20) < val)):  # or (100 < col < 170): Siva barva
                    cv2.circle(image, (x, y), r, (0, 0, 0), 4)
                    rref = r
                    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    BallPos = np.array([x, y])

                '''print(h[y][x])
                print(hue)
                print(sat)
                print(val)
                print(' ')

            else:
                print(h[y][x]*360/255)
                print(s[y][x]*100/255)
                print(v[y][x]*100/255)
                print('  ')'''

        # Določi referenčni center
        Center = np.array([int((GreenMarker[0] + BlueMarker[0] + RedMarker[0]) / 3),
                           int((GreenMarker[1] + BlueMarker[1] + RedMarker[1]) / 3)])
        # Center[1]=np.array(int((GreenMarker[1]+BlueMarker[1]+RedMarker[1])/3))
        cv2.circle(image, (Center[0], Center[1]), rref, (10, 100, 100), 4)
        cv2.rectangle(image, (Center[0] - 5, Center[1] - 5), (Center[0] + 5, Center[1] + 5), (0, 128, 255), -1)
        cv2.line(image, (Center[0], Center[1]), (BallPos[0], BallPos[1]), (255, 0, 0), 2)

        # Določi x in y osi ter enotski matriki
        '''Xos=np.array([(-BlueMarker[0]+GreenMarker[0]),((-BlueMarker[1]+GreenMarker[1]))])
        XosE=Xos/np.linalg.norm(Xos)

        Yos=np.array([Xos[1],-Xos[0]])
        YosE=Yos/np.linalg.norm(Yos)'''

        # Pretvorba iz px v mm
        kx = 260 / np.linalg.norm(Xos)

        ptmp = np.array([int((BlueMarker[0] + GreenMarker[0]) / 2), int((BlueMarker[1] + GreenMarker[1]) / 2)])
        cv2.rectangle(image, (ptmp[0] - 5, ptmp[1] - 5), (ptmp[0] + 5, ptmp[1] + 5), (0, 128, 255), -1)
        ytmp = np.array([(-ptmp[0] + RedMarker[0]), ((-ptmp[1] + RedMarker[1]))])
        ky = 200 / np.linalg.norm(ytmp)

        # Določanje rotacijske matrike met ks slike in ks robota
        # R=np.array([[XosE[0],XosE[1]],[YosE[0],YosE[1]]])
        # Vy=np.matmul(R,YosE)
        # Vx=np.matmul(R,XosE)

        # Določanje vektorja odmika, smer in absolutna vrednost
        Rnew = -BallPos + Center
        Rd = -Rnew + Rold
        Rf = Rnew - 5 * Rd
        RdifE = (Rf) / np.linalg.norm(Rf)
        RdiffG = np.matmul(R, RdifE)
        RdiffG.astype('float32')
        Absdiff = np.linalg.norm(Rf) * kx
        Rold = Rnew
        # print(Absdiff)
        # print(RdiffG)

        send_vals(RdiffG[0], RdiffG[1], Absdiff)
        # print("Sent: ({},{},{})".format(RdiffG[0],RdiffG[1],Absdiff))
        sys.stdout.flush()

    # show the output image
    cv2.imshow("output", image)
    print(time.time() - T)
    # cv2.imshow("output", blur)
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
