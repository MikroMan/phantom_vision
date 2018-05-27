import math
import socket
import struct
import sys
import time

client_ip = "192.168.65.94"
port = 25000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP


def send_vals(x, y):
    vals = (x, y, 4)
    packer = struct.Struct('f f f')
    bin_data = packer.pack(*vals)
    sock.sendto(bin_data, (client_ip, port))


# -15 -> 15 range


r = 0.09
while True:
    for i in range(0, 360, 2):
        x = r * math.cos(i * math.pi / 180)
        y = r * math.sin(i * math.pi / 180)
        send_vals(x, y)
        print("Sent: ({},{})".format(x, y))
        sys.stdout.flush()

        time.sleep(0.05)
