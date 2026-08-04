import mujoco
import mujoco.viewer

import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray 
import numpy as np 

xml_path= "/home/digital_twin/robots/mobile_aloha_sim/aloha_mujoco/aloha/meshes_mujoco/aloha_v1.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)

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

class AlohaSim():

    def __init__(self):
        master_joint_topic = "/master/joint_right"
        sim_joint_topic = "/sim/joint_right"
        sim_image_topic = "sim_cam/color/image_raw"
        object_pos_topic = "/object_positions"
        timer_period = 1.0/30 #30hz
        self.object_names = {
            "red_cube":      0,   
            "purple_capsule": 7,  
            "blue_sphere":   14,
            "green_cylinder": 21, 
            "orange_ellipsoid": 28, 
            "brown_dish":    35,  
        }

        rospy.Subscriber(master_joint_topic, JointState, self.joint_right_callback)
        rospy.Subscriber(object_pos_topic, Float64MultiArray, self.object_pos_callback)  

        self.joint_pub = rospy.Publisher(sim_joint_topic, JointState, queue_size=20)
        self.image_pub = rospy.Publisher(sim_image_topic, Image, queue_size = 10)

        self.renderer = mujoco.Renderer(model, 480, 640)
        
        self.timer = rospy.Timer(rospy.Duration(timer_period), self.timer_callback)

    def timer_callback(self, event):
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "sim_cam_f"
        msg.height = 480
        msg.width = 640
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step= 1920

        self.renderer.update_scene(data, camera="fr_dabai")
        rgb = self.renderer.render()
        msg.data = rgb.tobytes()

        self.image_pub.publish(msg)

    def joint_right_callback(self, msg):
        if(not viewer.is_running()):
            viewer.close()

        data.ctrl[8:] = msg2pos(msg)
        
        for _ in range(10):
            mujoco.mj_step(model, data)
        
        self.joint_pub.publish(pos2msg(data.qpos))

        viewer.sync()

    def object_pos_callback(self, msg): 
        arr = np.array(msg.data)
        if len(arr) < 12:
            rospy.logwarn(f"Expected 12 values, got {len(arr)}")
            return
        
        for i, (name, base_idx) in enumerate(self.object_names.items()):
            if i >= 6:
                break
            x, y = arr[2*i], arr[2*i + 1]
            data.qpos[base_idx + 0] = x
            data.qpos[base_idx + 1] = y   

        mujoco.mj_forward(model, data)


def main():
    rospy.init_node("simulation_node")
    sim = AlohaSim()
    rospy.spin()

main()