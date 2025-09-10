import rclpy

import time

import os
import sys
ros_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(ros_dir, "..")
utils_dir = os.path.join(parent_dir, "utils")
core_dir = os.path.join(parent_dir, "core")
fig_dir = os.path.join(parent_dir, 'figures')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if ros_dir not in sys.path:
    sys.path.insert(0, ros_dir)
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)
import json
config = json.load(open(os.path.join(parent_dir, "config.json")))

from ros_utils import msg_to_pcd
from behaviors import MainNode

def main():
    print("entered main")
    
    rclpy.init()
    node = MainNode()
    left_arm = [0.1, 0.5, 0.1, 0, 0, 0]
    right_arm = [0.3, -0.7, 0.3, 90, 0, 90]
    node.set_hands(l_goal = [1.0]*6)
    input("press anything to start, ctrl c to cancel")


    #initial distant positons
    print(f"{left_arm=}")
    print(f"{right_arm=}")
    node.send_arm_goal(
        left_arr = left_arm,
        right_arr = right_arm
    )
    node.set_hands(r_goal= [0.0, 0.0, 0.0, 1.0, 0.0, 1.0])

    left_arm[1] -= 0.3
    right_arm[1] += 0.05
    right_arm[2] -= 0.3

    #get in closer and make contact
    print(f"{left_arm=}")
    print(f"{right_arm=}")
    node.send_arm_goal(
        left_arr = left_arm,
        right_arr = right_arm
    )

    right_arm[0]+=0.2
    #move right hand forward
    print(f"{left_arm=}")
    print(f"{right_arm=}")
    node.send_arm_goal(
        left_arr = left_arm,
        right_arr = right_arm
    )

    right_arm[1]+= 0.1
    right_arm[2]+= 0.35
    # #move right arm up and in
    # print(f"{left_arm=}")
    # print(f"{right_arm=}")
    # node.send_arm_goal(
    #     left_arr = left_arm,
    #     right_arr = right_arm
    # )


    right_arm[0]-= 0.3
    right_arm[2]+= 0.2
    #move right arm back and up
    print(f"{left_arm=}")
    print(f"{right_arm=}")
    node.send_arm_goal(
        left_arr = left_arm,
        right_arr = right_arm
    )
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()