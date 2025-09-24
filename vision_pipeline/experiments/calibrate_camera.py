import time
import rclpy
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
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
    camera_node = RealSenseSubscriber("/realsense/left_hand")
    print("camera initialized")
    radius = 0.2
    fixed_x = 0.15
    fixed_pitch = 90
    fixed_yaw = 0
    arc_start = np.pi/6
    arc_end = 5*(np.pi/6)
    arc_span = arc_end-arc_start
    rotation_point = (0.25,0.1)
    min_height = 0.05
    step_size = 0.1
    while rclpy.ok():
        ys = []
        zs = []
        rolls = []
        for i in np.arange(0,1 + step_size, step_size):
            theta = (i * arc_span) + arc_start
            roll = np.degrees(theta)
            print(f"{theta=}")

            new_y = radius * np.cos(theta) + rotation_point[0]
            new_z = radius * np.sin(theta) + rotation_point[1]
            new_z = max(new_z, min_height)

            ys.append(new_y)
            zs.append(new_z)
            rolls.append(roll)
            l_arm_pose = [fixed_x, new_y, new_z, roll, fixed_pitch, fixed_yaw]
            behavior_node.send_arm_goal(left_arr=l_arm_pose)
            
            rgb, depth, info, pose = camera_node.get_data()
            if rgb is not None and depth is not None and pose is not None:
                cv2.imshow("rgb", rgb)
                print(f"{pose=}")
                cv2.waitKey(1)



        plt.scatter(ys, zs)
        for c1, c2, theta in zip(ys, zs, rolls):
            dc1 = np.cos(np.radians(theta)) * radius
            dc2 = np.sin(np.radians(theta)) * radius
            plt.plot([c1, c1-dc1], [c2, c2-dc2], color="red")
        plt.ylim(bottom=-1, top=1)
        plt.show()
    # rclpy.shutdown()
    

if __name__ == "__main__":
    main()