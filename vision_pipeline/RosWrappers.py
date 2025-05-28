#!/usr/bin/env python3
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from realsense2_camera_msgs.msg import RGBD
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge

import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)

from VisionPipeline import VisionPipe
from FoundationModels import OWLv2, SAM2_PC
from h12_controller_wrapper.controller import ArmController
from RealsenseInterface import RealSenseSubscriber



class ROS_VisionPipe(VisionPipe, Node):
    def __init__(self):
        super().__init__()
        self.head_sub = RealSenseSubscriber("head")
        self.LARM_sub = RealSenseSubscriber("left_hand")
        self.track_strings = []

    def update(self, sub_str="head", debug=False):
        if sub_str == "head":
            sub = self.head_sub
        elif sub_str == "left_hand":
            sub = self.LARM_sub
        else:
            raise ValueError("sub_str must be 'head' or 'left_hand'")
        rgb_img = sub.latest_rgb
        depth_img = sub.latest_depth
        intrinsics = sub.get_intrinsics()
        pose = [0,0,0,0,0,0]
        if rgb_img is None or depth_img is None or intrinsics is None:
            #print("No image received yet.")
            return False
        if len(self.track_strings) == 0:
            #print("No track strings provided.")
            return False

        return super().update(rgb_img, depth_img, self.track_strings, intrinsics, pose, debug=debug)

    def add_track_string(self, new_track_string):
        if isinstance(new_track_string, str) and new_track_string not in self.track_strings:
            self.track_strings.append(new_track_string)
        elif isinstance(new_track_string, list):
            [self.track_strings.append(x) for x in new_track_string if x not in self.track_strings]
        else:
            raise ValueError("track_string must be a string or a list of strings")





def TestSubscriber(args=None):
    rclpy.init(args=args)
    cams = sys.argv[1:] if len(sys.argv) > 1 else ['head', 'left_hand']

    # Instantiate subscribers and create display windows
    subs = []
    for cam in cams:
        sub = RealSenseSubscriber(cam)
        subs.append(sub)
        cv2.namedWindow(f"{cam}/RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"{cam}/Depth", cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            # Display images for each subscriber
            for sub in subs:
                rgb = sub.latest_rgb.copy() if sub.latest_rgb is not None else np.zeros((480, 640, 3), np.uint8)
                depth = sub.latest_depth.copy() if sub.latest_depth is not None else np.zeros((480, 640, 3), np.uint8)
                depth = depth.astype(np.float32)
                depth /= max(depth.max(), 1e-6)  # Normalize depth to [0, 1] for display
                depth = (depth * 255).astype(np.uint8)  # Scale to [0, 255] for display
                cv2.imshow(f"{sub.camera_key}/RGB", rgb)
                cv2.imshow(f"{sub.camera_key}/Depth", depth)

            # WaitKey for refresh and exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        for sub in subs:
            sub.shutdown()
        cv2.destroyAllWindows()
        rclpy.shutdown()


def TestFoundationModels(args=None):
    rclpy.init(args=args)
    sub = RealSenseSubscriber("head")
    OWL = OWLv2()
    SAM = SAM2_PC()
    while rclpy.ok():
        rgb_img = sub.latest_rgb
        depth_img = sub.latest_depth
        intrinsics = sub.get_intrinsics()

        if rgb_img is None or depth_img is None or intrinsics is None:
            print("Waiting for images...")
            continue
        print("RGB img shape: ", rgb_img.shape)
        print("Depth img shape: ", depth_img.shape)
        obs_pose = [0, 0, 0, 0, 0, 0]

        print("RGB img shape: ", rgb_img.shape)
        print("Depth img shape: ", depth_img.shape)
        querries = ["drill", "screw driver", "wrench"]
        predictions_2d = OWL.predict(rgb_img, querries, debug=False)
        for query_object, canditates in predictions_2d.items():
            #print("\n\n")
            point_clouds, boxes, scores,  rgb_masks, depth_masks = SAM.predict(rgb_img, depth_img, canditates["boxes"], canditates["scores"], intrinsics, debug=False)
            n = 5
            fig, axes = plt.subplots(5, 2, figsize=(20, 10))
            for i in range(min(n, len(point_clouds))):
                axes[i, 0].imshow(rgb_masks[i])
                axes[i, 1].imshow(depth_masks[i], cmap='gray')
                axes[i, 0].set_title(f"{query_object} {i} Score:{scores[i]:.2f}")
            fig.tight_layout()
            fig.suptitle(f"{query_object} RGB and Depth Masks")
            plt.show(block = False)
        plt.show(block = True)
    return None


def TestVisionPipe(args=None):
    rclpy.init(args=args)
    robot = ArmController('/home/humanoid/vp_ws/src/VisionPipeline/H12ControllerWrapper/assets/h1_2/h1_2.urdf',
                                   dt=0.01,
                                   vlim=1.0,
                                   visualize=True)
    robot.left_ee_target_pose = [1, 1, 1, 0, 0, 0]

    VP = ROS_VisionPipe()
    VP.add_track_string("drill")
    #VP.add_track_string(["wrench", "screwdriver"])
    success_counter = 0
    while success_counter < 5:
        success = VP.update()
        success_counter += 1 if success != False else 0
        if success:
            print(f"Success {success_counter} with {len(VP.tracked_objects)} tracked objects")
            #VP.vis_belief2D(query="drill", blocking=False, prefix = f"T={success_counter} ")
        if not success:
            print("No new predictions, waiting for next update...")

    print("Success counter: ", success_counter)

    for object, predictions in VP.tracked_objects.items():
        print(f"{object=}")
        print(f"   {len(predictions['boxes'])=}, {len(predictions['pcds'])=}, {predictions['scores'].shape=}")
        for i, pcd in enumerate(predictions["pcds"]):
            print(f"   {i=}, {predictions['scores'][i]=}")
        print(f"{object=}")
        VP.vis_belief2D(query=object, blocking=False, prefix=f"Final {object} predictions")
    print("call show")
    plt.show()
