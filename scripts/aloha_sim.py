# sim_server.py
import socket
import struct
import mujoco
import mujoco.viewer
import threading

import rospy
import rosnode
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image

# Load model
xml_path= "/home/digital_twin/robots/mobile_aloha_sim/aloha_mujoco/aloha/meshes_mujoco/aloha_v1.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

class AlohaSim(Node):

    def __init__(Node):

        master_joint_topic = "/master/joint_right"
        sim_joint_topic = "/sim/joint_right"
        sim_image_topic = "sim_cam/color/image_raw"
        timer_period = 1.0/30 #30hz

        rospy.Subscriber(master_joint_topic, JointState, self.joint_right_callback)
        self.joint_pub = rospy.Publisher(sim_joint_topic, JointState, queue_size=20)
        self.image_pub = rospy.Publisher(sim_image_topic, Image, queue_size = 10)
        self.timer = self.create_timer(timer_period, self.timer_callback)

def msg2pos(msg):
    position = list(msg.position)
    position[2] *= -1
    position[3], position[4] = position[4], position[3]
    position[6] /= 2
    position.append(position[6])
    return position

def pos2msg(pos):
    position = list(pos[-8:])
    position[2] *= -1
    position[3], position[4] = position[4], position[3]
    position[6] *=2
    position = position[:-1]

    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.position = position

    return msg

# def timer_callback(self):
#     msg = Image()
#     msg.header.stamp = rospy.Time.now()
#     msg.header.frame_id = "sim_cam_f"
#     msg.height = 480
#     msg.width = 640
#     msg.encoding = "rgb8"
#     msg.is_bigendian = False
#     msg.step= 1920
#     msg.data = []
#     image_pub.publish(msg)

def joint_right_callback(self, msg):
    if(not viewer.is_running()):
        viewer.close()
    position = msg2pos(msg)
    # print(position)

    data.ctrl[8:] = position

    for _ in range(10):
        mujoco.mj_step(model, data)
    
    joint_pub.publish(pos2msg(data.qpos))

    viewer.sync()

def main():
    rospy.init_node("simulation_node")

    timer = 
    rospy.spin()


main()
