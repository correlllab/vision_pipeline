import time
import rclpy
import numpy as np
import os
import sys
import cv2
exp_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(exp_dir, "..")
utils_dir = os.path.join(parent_dir, "utils")
core_dir = os.path.join(parent_dir, "core")
fig_dir = os.path.join(parent_dir, 'figures')
ros_dir = os.path.join(parent_dir, "ROS")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if ros_dir not in sys.path:
    sys.path.insert(0, ros_dir)
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)
if exp_dir not in sys.path:
    sys.path.insert(0, exp_dir)
from behaviors import MainNode
from RosRealsense import RealSenseSubscriber
def main():
    behavior_node = MainNode(vp = False)
    rclpy.init()
    camera_node = RealSenseSubscriber("/realsense/left_hand")
    print("camera initialized")
    radius = 0.6
    fixed_y = 0.2
    fixed_pitch = 0
    fixed_yaw = 0
    arc_span = np.pi / 2
    while rclpy.ok():
        for i in np.arange(0,1.1, 0.1):
            theta = i * arc_span
            roll = i * np.pi  # roll changes independently, optional

            new_x = radius * np.cos(theta)
            new_z = max(radius * np.sin(theta), 0.1)

            l_arm_pose = [new_x, fixed_y, new_z, roll, fixed_pitch, fixed_yaw]
            behavior_node.send_arm_goal(left_arr=l_arm_pose)

            time.sleep(0.1)  # allow time between goals


            
            rgb, depth, info, pose = camera_node.get_data()
            if rgb is not None and depth is not None and pose is not None:
                cv2.imshow("rgb", rgb)
                cv2.imshow("depth", depth / np.max(depth))
                print(f"{pose=}")
                cv2.waitKey(1)
            time.sleep(0.1)
    rclpy.shutdown()
    

if __name__ == "__main__":
    main()