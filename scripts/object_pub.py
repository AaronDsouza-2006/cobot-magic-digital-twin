#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray

def main():
    rospy.init_node("object_position_publisher")
    pub = rospy.Publisher("/object_positions", Float64MultiArray, queue_size=10)
    
    rate = rospy.Rate(10)  # 10 Hz
    
    # Table bounds (approximate based on your XML: center 0.7, 0.0, size 0.3 x 0.5)
    x_min, x_max = 0.45, 0.95
    y_min, y_max = -0.4, 0.4
    
    while not rospy.is_shutdown():
        msg = Float64MultiArray()
        
        # Generate 6 random (x, y) pairs = 12 values
        positions = []
        for _ in range(6):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            positions.extend([x, y])
        
        msg.data = positions
        pub.publish(msg)
        
        rospy.loginfo(f"Published positions: {positions}")
        rate.sleep()

if __name__ == "__main__":
    main()