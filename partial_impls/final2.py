import socket
import time
import struct
import cv2
import cv2 as cv
import numpy as np
import math

camera = cv2.VideoCapture(1)
RAD_REAL = 0.15
ref_center = None
ratioCoeff = None

client_ip = "192.168.65.94"
port = 25000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
pos_old = None

frameskip_counter = 0
trajectory_track = []

edge_pts = []

frame = None


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global edge_pts

    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        pts.append(np.asarray((x, y)))


def send_vals(x, y, z, ang):
    vals = (x, y, z, ang)
    packer = struct.Struct('f f f f')
    bin_data = packer.pack(*vals)
    sock.sendto(bin_data, (client_ip, port))


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

breakCapture = False

params = {}

while not breakCapture:
    for i in range(3):
        ret, frame = camera.read()

    cv.putText(frame, 'Double click 3 edge points to calibrate.', (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, 255)
    while len(edge_pts) < 3:

        cv2.imshow('image', frame)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('e'):
            break
    cv2.imshow('image', frame)

    center = sum(edge_pts) / 3
    cv2.circle(frame, (center[0].astype('int'), center[1].astype('int')), 4, (0, 0, 255), -1)
    radius = int(math.hypot(center[0] - edge_pts[0][0], center[1] - edge_pts[0][1]))

    circlesPlatform = cv2.HoughCircles(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, 1000, param1=100,
                                       param2=10, minRadius=radius - 30,
                                       maxRadius=radius + 30)

    if circlesPlatform is None:
        print('error')
    circlesPlatform = np.uint16(np.around(circlesPlatform))
    for ball in circlesPlatform[0, :]:
        cv2.circle(frame, (ball[0], ball[1]), ball[2], (0, 255, 0), 2)
        cv2.circle(frame, (ball[0], ball[1]), 2, (0, 0, 255), 3)

    balls = cv2.HoughCircles(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, 1000, param1=100,
                             param2=10, minRadius=int((radius - 30) / 13),
                             maxRadius=int((radius + 30) / 13))

    if balls is None:
        print('No ball')
    else:
        balls = np.uint16(np.around(balls))
        for ball in balls[0, :]:
            j = np.array([109, 250, 17])
            cv2.circle(frame, (ball[0], ball[1]), ball[2], (0, 255, 0), 2)
            cv2.circle(frame, (ball[0], ball[1]), 2, (0, 0, 255), 3)
            mask = np.zeros(frame.shape, dtype=np.uint8)
            cv2.circle(mask, (ball[0], ball[1]), ball[2], (255, 255, 255), -1, 8, 0)

    cv.putText(frame, 'Press: y - continue, n - repeat.', (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, 255)
    cv2.imshow('image', frame)

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == ord('n'):
            edge_pts = []
            break
        elif k == ord('y'):
            breakCapture = True

            params = {
                "platform_min": radius - 30,
                "platform_max": radius + 30,
                "ball_min": (radius - 30) / 13,
                "ball_max": (radius + 30) / 13
            }
            print(params)

            ref_center = circlesPlatform[0, 0]
            ratioCoeff = RAD_REAL / ref_center[2].astype('float')

            break

    cv2.imshow('image', frame)

# while True:
#     ret, frame = camera.read()
#     pic_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     circlesPlatform = cv.HoughCircles(pic_gray, cv.HOUGH_GRADIENT, 1, 1000, param1=100, param2=10, minRadius=180,
#                                       maxRadius=230)
#     if circlesPlatform is None:
#         continue
#     circlesPlatform = np.uint16(np.around(circlesPlatform))
#     for i in circlesPlatform[0, :]:
#         cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
#         cv.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
#
#     cv.putText(frame, 'Is detection OK? Press Y/N.', (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, 255)
#     cv.imshow('Capture platform', frame)
#
#     if cv.waitKey() == 121:  # is yes pressed
#         ref_center = circlesPlatform[0, 0]
#         ratioCoeff = RAD_REAL / ref_center[2].astype('float')
#
#         print('Captured reference platform position')
#         break


cv.destroyAllWindows()
print("Starting ball detection")


def find_closest_circle(circles):
    circles = np.uint16(np.around(circles))
    min_dist = None
    min_ball = (1000000, 1000000, 100000)

    for ball in circles[0, :]:
        dist = math.hypot(ref_center[0].astype('float') - ball[0].astype('float'),
                          ref_center[1].astype('float') - ball[1].astype('float'))

        if (min_dist is None or dist < min_dist) and dist < ref_center[2]:
            # print('Min',dist*ratioCoeff)
            if dist * ratioCoeff > 0.13:
                print(frameskip_counter, 'Skip outside of range', dist * ratioCoeff)
                continue
            min_dist = dist
            min_ball = ball
    if min_dist is None:
        return None
    else:
        return min_ball


def process(min_ball):
    global pos_old, frameskip_counter, trajectory_track

    trajec.append(min_ball)
    if len(trajec) > 100:
        trajec.pop(-1)

    dist_real = ratioCoeff * (math.hypot(min_ball[0].astype(float) - ref_center[0].astype(float),
                                         min_ball[1].astype(float) - ref_center[1].astype(float)))
    y_rot = -ratioCoeff * (min_ball[0].astype(float) - ref_center[0].astype(float))
    x_rot = -ratioCoeff * (min_ball[1].astype(float) - ref_center[1].astype(float))

    pos_new = np.array([x_rot, y_rot])
    pos_diff = pos_new - np.asarray([old_ball[0], old_ball[1]])

    if pos_diff[0] == 0.0 and pos_diff[1] == 0.0:
        pos_diff = np.asarray([0.0000001, pos_diff[1]])

    if pos_new[0] == 0.0 and pos_new[1] == 0.0:
        pos_new = np.asarray([0.0000001, pos_new[1]])

    cos_al = np.dot(pos_diff, pos_new) / (np.linalg.norm(pos_diff) * np.linalg.norm(pos_new))
    vel_rad = (cos_al * pos_diff * 30) / ((frameskip_counter + 2) / 2)
    if np.isnan(vel_rad[0]) or np.isnan(vel_rad[1]):
        vel_rad = np.array([0, 0])

    dir_vec = pos_new + 3 * np.asarray(pos_diff)

    if dir_vec[0] == 0.0 and dir_vec[1] == 0.0:
        dir_vec = np.asarray([0.0000001, dir_vec[1]])

    rot_axis = np.cross(np.array([0, 0, 1]), dir_vec)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    sign = -1 if dist_real > math.hypot(old_ball[0], old_ball[1]) else 1

    angle = - 0.7 * dist_real + 0.3 * np.linalg.norm(vel_rad) * sign

    old_ball = pos_new
    if math.degrees(angle) > 8:
        angle = math.radians(8)
    frameskip_counter = 0

    send_vals(rot_axis[0], rot_axis[1], rot_axis[2], angle)


def is_holder(ball_pos, pic):
    if ball_pos is None:
        return True

    center = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[ball_pos[1], ball_pos[0]]

    return True if center < 40 or center > 80 else False


def render(frame):
    global ref_center
    # narisi platformo
    cv.circle(frame, (ref_center[0], ref_center[1]), ref_center[2], (0, 255, 0), 2)
    cv.circle(frame, (ref_center[0], ref_center[1]), 2, (0, 0, 255), 3)

    for pos in trajectory_track:
        cv.circle(frame, (ref_center[0], ref_center[1]), ref_center[2], (0, 255, 0), 2)
        cv.circle(frame, (pos[0], pos[1]), 2, (0, 0, 255), 3)

    cv.imshow('Detekcija zoge', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True


if __name__ == '__main__':
    while True:
        ret, frame = camera.read()
        pic_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # 1, 60, param1=100, param2=10, minRadius=7, maxRadius=20

        circlesBall = cv.HoughCircles(pic_gray, cv.HOUGH_GRADIENT, 1, 60, param1=100, param2=10,
                                      minRadius=int(params['ball_min']),
                                      maxRadius=int(params['ball_max']))

        if circlesBall is None:
            frameskip_counter += 1
            render(frame)
            # print(skipped_frames, 'Frame skipped - no balls')
            continue

        for min_ball in circlesBall[:, 0]:
            cv.circle(frame, (min_ball[0], min_ball[1]), min_ball[2], (0, 255, 0), 2)
            cv.circle(frame, (min_ball[0], min_ball[1]), 2, (0, 0, 255), 3)

        min_ball = find_closest_circle(circlesBall)

        if min_ball is None or is_holder(min_ball, frame):
            render(frame)
            print(frameskip_counter, 'Frame skipped - no min ball')
            frameskip_counter += 1
            continue

        if pos_old is None:
            x_real = ratioCoeff * (min_ball[0].astype(float) - ref_center[0].astype(float))
            y_real = ratioCoeff * (min_ball[1].astype(float) - ref_center[1].astype(float))
            pos_old = np.asarray([-y_real, -x_real])

        # narise zogico
        cv.circle(frame, (min_ball[0], min_ball[1]), min_ball[2], (0, 255, 0), 2)
        cv.circle(frame, (min_ball[0], min_ball[1]), 2, (0, 0, 255), 3)

        process(min_ball)

        loc_str = "Ball: ({:.3f},{:.3f}) - {:.3f}".format(pos_old[0] * 1000, pos_old[1] * 1000,
                                                          math.hypot(pos_old[0], pos_old[1]) * 1000)
        cv.putText(frame, loc_str, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, 200, thickness=2)

        render(frame)

    cv.destroyAllWindows()
