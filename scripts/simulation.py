# sim_server.py
import socket
import struct
import mujoco
import mujoco.viewer
import threading

import rospy
import rosnode
from sensor_msgs.msg import JointState


# Load model
xml_path= "/home/digital_twin/robots/mobile_aloha_sim/aloha_mujoco/aloha/meshes_mujoco/aloha_v1.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# # Socket setup
# UDP_IP = "127.0.0.1"
# UDP_PORT = 5005
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.setblocking(False)  # Non-blocking
# sock.bind((UDP_IP, UDP_PORT))

# Shared state
latest_angles = [0.0]  # Default position
# lock = threading.Lock()

# def receiver_thread():
#     global latest_angles
#     while True:
#         try:
#             packet, _ = sock.recvfrom(1024)
#             # Unpack: one float (4 bytes)
#             angle = struct.unpack('f', packet)[0]
#             with lock:
#                 latest_angles[0] = angle
#         except BlockingIOError:
#             pass  # No data available

# # Start receiver in background
# threading.Thread(target=receiver_thread, daemon=True).start()

# Main simulation loop with viewer
viewer = mujoco.viewer.launch_passive(model, data)
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         # Apply latest received control
#         with lock:
#             data.ctrl[0] = latest_angles[0]
        
#         # Step physics
#         for _ in range(60):
#             mujoco.mj_step(model, data)
        
#         viewer.sync()

def joint_right_callback(msg):
    if(not viewer.is_running()):
        viewer.close()
    position = list(msg.position)
    print(position)
    position[1] -= 0.785
    position[2] += 1.57
    position[4]
    position[6] /= 2
    position.append(position[6])

    data.ctrl[8:] = position


    for _ in range(60):
        mujoco.mj_step(model, data)
    
    viewer.sync()

def main():
    rospy.init_node("simulation_control_node")

    joint_right_topic = "/master/joint_right"

    rospy.Subscriber(joint_right_topic, JointState, joint_right_callback)

    rospy.spin()

main()
