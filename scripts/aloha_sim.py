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

viewer = mujoco.viewer.launch_passive(model, data)

def joint_right_callback(msg):
    if(not viewer.is_running()):
        viewer.close()
    position = list(msg.position)

    position[2] *= -1
    position[3], position[4] = position[4], position[3]
    position[6] /= 2
    position.append(position[6])
    print(position)

    data.ctrl[8:] = position

    for _ in range(10):
        mujoco.mj_step(model, data)
    
    viewer.sync()

def main():
    rospy.init_node("simulation_control_node")

    joint_right_topic = "/master/joint_right"

    rospy.Subscriber(joint_right_topic, JointState, joint_right_callback)

    rospy.spin()

main()
