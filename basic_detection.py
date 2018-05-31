"""
Authors: Robi Ravnikar, Tine Subic
Faculty of Electrotechnics, 2017/18
Robotic vision

The program is used in conjunction with Phantom robots and corresponding Simulink code for motion tracking and ball
balancing on a platform controlled by 3 Parallel Phantom robots via UDP communication
"""

import socket
import time
import struct
import cv2 as cv
import numpy as np
import math
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Control Phantom robots via machine vision.')
    parser.add_argument('-x', '--xpc-host', default='192.168.65.94',
                        help='Host IP address for XPC target')
    parser.add_argument('-p', '--port', type=int, default=25000, help='Port for XPC target')
    parser.add_argument('-c', '--camera-id', type=int, default=1, help='Camera ID. 1 if secondary, 0 if primary')
    parser.add_argument('-r', '--radius', type=float, default=0.15, help='Platform radius in meters.')

    args = parser.parse_args()
    print("Parsed args. Platform={}cm, XPC={}:{}".format(args.radius * 100, args.xpc_host, args.port))
    return args


print('Program started...')
args = parse_args()

camera = cv.VideoCapture(args.camera_id)
ref_center = None
ratioCoeff = None
pos_old = None
trajectory = []
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
frameskip_counter = 0


def send_vals(x, y, z, ang):
    """Send 4x32b decimal values via UDP"""
    data = (x, y, z, ang)
    packer = struct.Struct('f f f f')
    bin_data = packer.pack(*data)
    sock.sendto(bin_data, (args.xpc_host, args.port))


print('Starting platform detection...')
while True:
    ret, frame = camera.read()
    pic_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    circlesPlatform = cv.HoughCircles(pic_gray, cv.HOUGH_GRADIENT, 1, 1000, param1=100, param2=10, minRadius=180,
                                      maxRadius=230)
    if circlesPlatform is None:
        continue
    circlesPlatform = np.uint16(np.around(circlesPlatform))
    for i in circlesPlatform[0, :]:
        cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv.putText(frame, 'Is detection OK? Press Y/N.', (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv.imshow('Capture platform', frame)

    if cv.waitKey() == 121:  # is yes pressed
        ref_center = circlesPlatform[0, 0]
        ratioCoeff = args.radius / ref_center[2].astype('float')

        print('Captured reference platform position')
        break

cv.destroyAllWindows()
print("Starting motion tracking")


def find_closest_circle(circles):
    """Searches for ball closest for center - this is interpreted as balanced object"""
    circles = np.uint16(np.around(circles))
    min_dist = None
    ball = (1000000, 1000000, 100000)

    for circle in circles[0, :]:
        # calculate distance from center
        dist = math.hypot(ref_center[0].astype('float') - circle[0].astype('float'),
                          ref_center[1].astype('float') - circle[1].astype('float'))

        # find minimum and skip any that are inside dead zone - outer edge of platform
        if (min_dist is None or dist < min_dist) and dist < ref_center[2]:
            # print('Min',dist*ratioCoeff)
            if dist * ratioCoeff > 0.13:
                print(frameskip_counter, 'Skip outside of range', dist * ratioCoeff)
                continue
            min_dist = dist
            ball = circle

    return None if min_dist is None else ball


def process(min_ball):
    global pos_old, frameskip_counter, trajectory

    # store 100 last positions of trajectory
    trajectory.insert(0, min_ball)
    if len(trajectory) >= 100:
        trajectory.pop(-1)

    # convert ball position into robot coords
    dist = ratioCoeff * (math.hypot(min_ball[0].astype(float) - ref_center[0].astype(float),
                                    min_ball[1].astype(float) - ref_center[1].astype(float)))
    y_rot = -ratioCoeff * (min_ball[0].astype(float) - ref_center[0].astype(float))
    x_rot = -ratioCoeff * (min_ball[1].astype(float) - ref_center[1].astype(float))

    pos_new = np.array([x_rot, y_rot])
    pos_diff = pos_new - np.asarray([old_ball[0], old_ball[1]])

    # check for 0.0 values to avoid division by 0
    if pos_diff[0] == 0.0 and pos_diff[1] == 0.0:
        pos_diff = np.asarray([0.0000001, pos_diff[1]])

    if pos_new[0] == 0.0 and pos_new[1] == 0.0:
        pos_new = np.asarray([0.0000001, pos_new[1]])

    cosine_alpha = np.dot(pos_diff, pos_new) / (np.linalg.norm(pos_diff) * np.linalg.norm(pos_new))
    v_radial = cosine_alpha * pos_diff * 30
    # default to (0,0) if singularity or invalid values
    if np.isnan(v_radial[0]) or np.isnan(v_radial[1]):
        v_radial = np.array([0, 0])

    poz_vec = pos_new + 3 * np.asarray(pos_diff)

    if poz_vec[0] == 0.0 and poz_vec[1] == 0.0:
        poz_vec = np.asarray([0.0000001, poz_vec[1]])

    rot_axis = np.cross(np.array([0, 0, 1]), poz_vec)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    # define speed sign, towards or away from  center
    sign = -1 if dist > math.hypot(old_ball[0], old_ball[1]) else 1

    # calculate angle
    angle = - 0.7 * dist + 0.3 * np.linalg.norm(v_radial) * sign

    # store position for next frame
    old_ball = pos_new
    # limit angle to 8 degrees
    if (math.degrees(angle)) > 8.0:
        angle = math.radians(8.0)

    send_vals(rot_axis[0], rot_axis[1], rot_axis[2], angle)
    frameskip_counter = 0


def is_holder(ball_pos, pic):
    """Check if detected ball is mis-detected platform holder based on center intensity"""
    if ball_pos is None:
        return True
    # swapped coordinates due to opencv inconsistencies
    center = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)[ball_pos[1], ball_pos[0]]

    print(center < 40 or center > 80)
    return True if center < 40 or center > 80 else False


def render(frame):
    """Render frame with displayed platform"""
    global ref_center, trajectory
    # narisi platformo
    cv.circle(frame, (ref_center[0], ref_center[1]), ref_center[2], (0, 255, 0), 2)
    cv.circle(frame, (ref_center[0], ref_center[1]), 2, (0, 0, 255), 3)

    for pos in trajectory:
        cv.circle(frame, (pos[0], ref_center[1]), 2, (0, 0, 255), 2)

    cv.imshow('Detekcija zoge', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        return True


if __name__ == '__main__':
    while True:
        ret, frame = camera.read()
        pic_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        circlesBall = cv.HoughCircles(pic_gray, cv.HOUGH_GRADIENT, 1, 60, param1=100, param2=10, minRadius=15,
                                      maxRadius=20)

        # skip processing if no balls detected, or only platform holder is detected
        if circlesBall is None:
            frameskip_counter += 1
            render(frame)
            # print(skipped_frames, 'Frame skipped - no balls')
            continue

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
