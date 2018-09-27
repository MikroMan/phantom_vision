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

camera = cv.VideoCapture(1)

RAD_REAL = 0.15

client_ip = "192.168.65.94"
port = 25000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP

# reference data
ref_center = None
ratioCoeff = None
pos_old = None
frame = None

# skipping counter for movement smoothing
frameskip_counter = 0

# old ball positions for drawing trajectory on screen
trajectory_track = []

# storage for calibration edge points
edge_pts = []


# mouse callback function
def draw_circle(event, x, y, flags, param):
    """Triggered on doubleclick for param calibration"""
    global edge_pts

    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(frame, (x, y), 4, (255, 0, 0), -1)
        edge_pts.append(np.asarray((x, y)))


def send_vals(x, y, z, ang):
    """Send 4x32b values to UDP socket"""
    vals = (x, y, z, ang)
    packer = struct.Struct('f f f f')
    bin_data = packer.pack(*vals)
    sock.sendto(bin_data, (client_ip, port))


cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)

breakCapture = False

params = {}

# run until user confirms detection with pressing y
while not breakCapture:
    for i in range(3):
        # skip a few frames to avoid occasional camera buffer lag or slow focus
        ret, frame = camera.read()

    cv.putText(frame, 'Double click 3 edge points to calibrate.', (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, 255,
               thickness=2)

    # capture 3 points on edge
    while len(edge_pts) < 3:
        cv.imshow('image', frame)
        k = cv.waitKey(20) & 0xFF
        if k == ord('e'):
            break
    cv.imshow('image', frame)

    # calculate center position and radius of platform
    center = sum(edge_pts) / 3
    cv.circle(frame, (center[0].astype('int'), center[1].astype('int')), 4, (0, 0, 255), -1)
    radius = int(math.hypot(center[0] - edge_pts[0][0], center[1] - edge_pts[0][1]))

    circlesPlatform = cv.HoughCircles(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.HOUGH_GRADIENT, 1, 1000, param1=100,
                                      param2=10, minRadius=radius - 30,
                                      maxRadius=radius + 30)

    if circlesPlatform is None:
        print('error')
    circlesPlatform = np.uint16(np.around(circlesPlatform))
    for ball in circlesPlatform[0, :]:
        cv.circle(frame, (ball[0], ball[1]), ball[2], (0, 255, 0), 2)
        cv.circle(frame, (ball[0], ball[1]), 2, (0, 0, 255), 3)

    # detect ball based on platform to ball ratio
    balls = cv.HoughCircles(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.HOUGH_GRADIENT, 1, 1000, param1=100,
                            param2=10, minRadius=int((radius - 30) / 13),
                            maxRadius=int((radius + 30) / 13))

    if balls is None:
        print('No ball')
    else:
        balls = np.uint16(np.around(balls))
        for ball in balls[0, :]:
            j = np.array([109, 250, 17])
            cv.circle(frame, (ball[0], ball[1]), ball[2], (0, 255, 0), 2)
            cv.circle(frame, (ball[0], ball[1]), 2, (0, 0, 255), 3)
            mask = np.zeros(frame.shape, dtype=np.uint8)
            cv.circle(mask, (ball[0], ball[1]), ball[2], (255, 255, 255), -1, 8, 0)

    cv.putText(frame, 'Press: y - continue, n - repeat.', (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.9, 255, thickness=2)
    cv.imshow('image', frame)

    # wait for user interaction, if presses y then continue to program, else repeat procedure
    while True:
        k = cv.waitKey(20) & 0xFF
        if k == ord('n'):
            edge_pts = []
            break
        elif k == ord('y'):
            breakCapture = True

            # store params for later detector
            params = {
                "platform_min": radius - 30,
                "platform_max": radius + 30,
                "ball_min": (radius - 30) / 13,
                "ball_max": (radius + 30) / 13
            }

            ref_center = circlesPlatform[0, 0]
            ratioCoeff = RAD_REAL / ref_center[2].astype('float')

            break

    cv.imshow('image', frame)

print("Starting ball detection")


def find_closest_circle(circles):
    """Find closest circle according to distance from platform center"""
    circles = np.uint16(np.around(circles))
    min_dist = None
    min_ball = (1000000, 1000000, 100000)

    for ball in circles[0, :]:
        dist = math.hypot(ref_center[0].astype('float') - ball[0].astype('float'),
                          ref_center[1].astype('float') - ball[1].astype('float'))

        # skip objects too far or outside platform
        if (min_dist is None or dist < min_dist) and dist < ref_center[2]:

            # deadzone on the outer edge of platform to avoid straining robots
            if dist * ratioCoeff > 0.13:
                print(frameskip_counter, 'Skip outside of range', dist * ratioCoeff)
                continue
            min_dist = dist
            min_ball = ball
    return None if min_dist is None else min_ball


def process(min_ball):
    """Processes ball position to calculate needed platform rotation to keep ball on it"""
    global pos_old, frameskip_counter, trajectory_track

    # keeping last 100 frames (approx 3-4 seconds of movement)
    trajectory_track.insert(0, min_ball)
    if len(trajectory_track) > 100:
        trajectory_track.pop(-1)

    # convert coordinates from camera system to robot coord system (px to cm)
    dist_real = ratioCoeff * (math.hypot(min_ball[0].astype(float) - ref_center[0].astype(float),
                                         min_ball[1].astype(float) - ref_center[1].astype(float)))
    y_rot = -ratioCoeff * (min_ball[0].astype(float) - ref_center[0].astype(float))
    x_rot = -ratioCoeff * (min_ball[1].astype(float) - ref_center[1].astype(float))

    pos_new = np.array([x_rot, y_rot])

    if pos_old is None:
        pos_old = pos_new
        return

    # movement direction vector from old to new position
    pos_diff = pos_new - np.asarray([pos_old[0], pos_old[1]])

    # validity checks to avoid singularity = infinite/nan values
    if pos_diff[0] == 0.0 and pos_diff[1] == 0.0:
        pos_diff = np.asarray([0.0000001, pos_diff[1]])

    if pos_new[0] == 0.0 and pos_new[1] == 0.0:
        pos_new = np.asarray([0.0000001, pos_new[1]])

    # calculate radial speed and check from invalid speeds (ex. no movement causes division by 0)
    cos_al = np.dot(pos_diff, pos_new) / (np.linalg.norm(pos_diff) * np.linalg.norm(pos_new))
    vel_rad = (cos_al * pos_diff * 30) / ((frameskip_counter + 2) / 2)  # smoothing on frameskip to avoid straining motors
    if np.isnan(vel_rad[0]) or np.isnan(vel_rad[1]):
        vel_rad = np.array([0, 0])

    dir_vec = pos_new + 3 * np.asarray(pos_diff)

    if dir_vec[0] == 0.0 and dir_vec[1] == 0.0:
        dir_vec = np.asarray([0.0000001, dir_vec[1]])

    # calculate axis along which we will rotate platform to direct ball towards center
    rot_axis = np.cross(np.array([0, 0, 1]), dir_vec)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    # check if ball is moving towards or away from center
    sign = -1 if dist_real > math.hypot(pos_old[0], pos_old[1]) else 1

    # PD regulator for rotation angle calucalteion
    angle = - 0.7 * dist_real + 0.3 * np.linalg.norm(vel_rad) * sign

    pos_old = pos_new

    # limit rotation angle to 8 degrees
    if math.degrees(angle) > 8:
        angle = math.radians(8)
    frameskip_counter = 0

    send_vals(rot_axis[0], rot_axis[1], rot_axis[2], angle)


def is_holder(ball_pos):
    """Check if detected ball is platform holder mount based on intensity"""
    if ball_pos is None:
        return True

    center = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)[ball_pos[1], ball_pos[0]]

    return True if center < 40 or center > 80 else False


def render(frame):
    """Render current frame with platform position and previous ball trajcetory in red"""
    global ref_center
    # narisi platformo
    cv.circle(frame, (ref_center[0], ref_center[1]), ref_center[2], (0, 255, 0), 2)
    cv.circle(frame, (ref_center[0], ref_center[1]), 2, (0, 0, 255), 3)

    prev = None
    for pos in trajectory_track:
        if prev is not None:
            # connect positions with lines to make it prettier
            cv.line(frame, (prev[0], prev[1]), (pos[0], pos[1]), (0, 0, 255), thickness=2, lineType=8)
        prev = pos
    cv.imshow('image', frame)

    cv.waitKey(1)


if __name__ == '__main__':
    """Continuously capture, detect and control ball position"""
    while True:
        ret, frame = camera.read()
        pic_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        circlesBall = cv.HoughCircles(pic_gray, cv.HOUGH_GRADIENT, 1, 60, param1=100, param2=10,
                                      minRadius=int(params['ball_min']),
                                      maxRadius=int(params['ball_max']))

        # skip frame if ball not found. Handle this issue later
        if circlesBall is None:
            frameskip_counter += 1
            render(frame)
            # print(skipped_frames, 'Frame skipped - no balls')
            continue

        for min_ball in circlesBall[:, 0]:
            cv.circle(frame, (min_ball[0], min_ball[1]), min_ball[2], (0, 255, 0), 2)
            cv.circle(frame, (min_ball[0], min_ball[1]), 2, (0, 0, 255), 3)

        min_ball = find_closest_circle(circlesBall)

        # skip frame if no ball or is invalid
        if min_ball is None or is_holder(min_ball):
            render(frame)
            print(frameskip_counter, 'Frame skipped - no min ball')
            frameskip_counter += 1
            continue

        # narise zogico
        cv.circle(frame, (min_ball[0], min_ball[1]), min_ball[2], (0, 255, 0), 2)
        cv.circle(frame, (min_ball[0], min_ball[1]), 2, (0, 0, 255), 3)

        process(min_ball)

        # display realtime position and distance from center
        loc_str = "Ball: ({:.3f},{:.3f}) - {:.3f}".format(min_ball[0] * 1000, min_ball[1] * 1000,
                                                          math.hypot(min_ball[0], min_ball[1]) * 1000)
        cv.putText(frame, loc_str, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, 200, thickness=2)

        render(frame)

    cv.destroyAllWindows()
