# controller_client.py
import socket
import struct
import time
import math
from rospy.node import Node
from sensor_msgs.msg import JointState

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send oscillating angle
t = 0
while True:
    angle = math.sin(t) * 1.0  # Oscillate between -1 and 1 radian
    packet = struct.pack('f', angle)
    sock.sendto(packet, (UDP_IP, UDP_PORT))
    print(f"Sent: {angle:.3f}")
    t += 0.05
    time.sleep(0.05)  # 20 Hz