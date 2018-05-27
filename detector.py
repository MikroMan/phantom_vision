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
    parser.add_argument('-c', '--camera-id', type=int, default=0, help='Camera ID. 1 if secondary, 0 if primary')
    parser.add_argument('-r', '--radius', type=float, default=0.15, help='Platform radius in meters.')

    args = parser.parse_args()
    print("Parsed args. Platform={}cm, XPC={}:{}".format(args.radius*100, args.xpc_host, args.port))
    return args


print('Program started...')
args = parse_args()



camera = cv.VideoCapture(args.camera_id)
ref_center = None
ratioCoeff = None
old_ball = None
trajectory = []
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP

frameskip_counter = 0


def send_vals(x, y, z, ang):
    """Send 4x32b decimal values via UDP"""
    vals = (x, y, z, ang)
    packer = struct.Struct('f f f f')
    bin_data = packer.pack(*vals)
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
    min_ball = (1000000, 1000000, 100000)

    for i in circles[0, :]:
        # calculate distance from center
        dist = math.hypot(ref_center[0].astype('float') - i[0].astype('float'),
                          ref_center[1].astype('float') - i[1].astype('float'))

        # find minimum and skip any that are inside dead zone - outer edge of platform
        if (min_dist is None or dist < min_dist) and dist < ref_center[2]:
            # print('Min',dist*ratioCoeff)
            if dist * ratioCoeff > 0.13:
                print(frameskip_counter, 'Skip outside of range', dist * ratioCoeff)
                continue
            min_dist = dist
            min_ball = i
    if min_dist is None:
        return None
    else:
        return min_ball


def process(min_ball):
    global old_ball, frameskip_counter, trajectory


    trajectory.insert(0, min_ball)
    if len(trajectory) >= 100:
        trajectory.pop(-1)


    dist_real = ratioCoeff * (math.hypot(min_ball[0].astype(float) - ref_center[0].astype(float),
                                         min_ball[1].astype(float) - ref_center[1].astype(float)))
    y_rot = -ratioCoeff * (min_ball[0].astype(float) - ref_center[0].astype(float))
    x_rot = -ratioCoeff * (min_ball[1].astype(float) - ref_center[1].astype(float))

    diff_vec = np.asarray([x_rot - old_ball[0], y_rot - old_ball[1]])

    vec_new = np.array([x_rot, y_rot])

    if diff_vec[0] == 0.0 and diff_vec[1] == 0.0:
        diff_vec = np.asarray([0.0000001, diff_vec[1]])

    if vec_new[0] == 0.0 and vec_new[1] == 0.0:
        vec_new = np.asarray([0.0000001, vec_new[1]])

    cos_al = np.dot(diff_vec, vec_new) / (np.linalg.norm(diff_vec) * np.linalg.norm(vec_new))
    vel_rad = cos_al * diff_vec * 30
    if np.isnan(vel_rad[0]) or np.isnan(vel_rad[1]):
        vel_rad = np.array([0, 0])

    poz_vec = np.asarray([x_rot, y_rot]) + 3 * np.asarray(diff_vec)

    if poz_vec[0] == 0.0 and poz_vec[1] == 0.0:
        poz_vec = np.asarray([0.0000001, poz_vec[1]])

    os_rot = np.cross(np.array([0, 0, 1]), poz_vec)
    os_rot = os_rot / np.linalg.norm(os_rot)

    if dist_real > math.hypot(old_ball[0], old_ball[1]):
        sign = -1
    else:
        sign = 1

    kot_rot = - 0.7 * dist_real + 0.3 * np.linalg.norm(vel_rad) * sign

    old_ball = np.asarray([x_rot, y_rot])
    if (kot_rot * 180 / np.pi) > 8:
        kot_rot = 8 * np.pi / 180
    frameskip_counter = 0

    send_vals(os_rot[0], os_rot[1], os_rot[2], kot_rot)


def is_holder(ball_pos, pic):
    """Check if detected ball is mis-detected platform holder based on center intensity"""
    if ball_pos is None:
        return True

    # swapped coordinates due to opencv inconsistencies
    center = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)[ball_pos[1], ball_pos[0]]

    if center < 40 or center > 80:
        # print('Holder detected! - ', center)
        return True
    # print('Ball detected - ', center)
    return False


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

        if old_ball is None:
            x_real = ratioCoeff * (min_ball[0].astype(float) - ref_center[0].astype(float))
            y_real = ratioCoeff * (min_ball[1].astype(float) - ref_center[1].astype(float))
            old_ball = np.asarray([-y_real, -x_real])

        # narise zogico
        cv.circle(frame, (min_ball[0], min_ball[1]), min_ball[2], (0, 255, 0), 2)
        cv.circle(frame, (min_ball[0], min_ball[1]), 2, (0, 0, 255), 3)

        process(min_ball)

        loc_str = "Ball: ({:.3f},{:.3f}) - {:.3f}".format(old_ball[0] * 1000, old_ball[1] * 1000,
                                                          math.hypot(old_ball[0], old_ball[1]) * 1000)
        cv.putText(frame, loc_str, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, 200, thickness=2)

        render(frame)

    cv.destroyAllWindows()
