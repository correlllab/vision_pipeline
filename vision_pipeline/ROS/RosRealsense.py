#!/usr/bin/env python3
import sys
import threading
from cv_bridge import CvBridge
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, Image


import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import sys

import time

import open3d as o3d
"""
Cheat Imports
"""
ros_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(ros_dir, "..")
utils_dir = os.path.join(parent_dir, "utils")
core_dir = os.path.join(parent_dir, "core")
fig_dir = os.path.join(parent_dir, 'figures')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from config import config
if ros_dir not in sys.path:
    sys.path.insert(0, ros_dir)
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)



#from SAM2 import SAM2_PC
#from BBBackBones import Gemini_BB, OWLv2
from ros_utils import decode_compressed_depth_image, TFHandler
from math_utils import pose_to_matrix, matrix_to_pose, display_2dCandidates



class RealSenseSubscriber(Node):
    def __init__(self, camera_name_space):
        print("Initializing RealSenseSubscriber for camera:", camera_name_space)
        node_name = f"{camera_name_space.replace(' ', '_').replace('/', '')}_subscriber"
        super().__init__(node_name)
        self.camera_name_space = camera_name_space
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_info = None
        self.latest_pose = None
        self._lock = threading.Lock()

        # TF2 buffer and listener with longer cache time
        self.tf_handler = TFHandler(self)

        # QoS for sensor data (Best effort + TRANSIENT_LOCAL)
        sensor_data_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriptions
        self.create_subscription(
            CompressedImage,
            f"{camera_name_space}/color/image_raw/compressed",
            self._rgb_callback,
            sensor_data_qos,
        )
        self.create_subscription(
            CompressedImage,
            f"{camera_name_space}/aligned_depth_to_color/image_raw/compressedDepth",
            self._depth_callback,
            sensor_data_qos,
        )
        self.create_subscription(
            CameraInfo,
            f"{camera_name_space}/color/camera_info",
            self._info_callback,
            sensor_data_qos,
        )

        # Private executor & spin thread for this node
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._spin_thread.start()

    def _rgb_callback(self, msg: CompressedImage):
        try:
            rgb_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            with self._lock:
                # print("[REALSENSE SUB rgb] lock aquired")
                self.latest_rgb = rgb_img
                # print("[REALSENSE SUB rgb] lock released")

        except Exception as e:
            print(f"Error processing RGB image for {self.camera_name}: {e}")

    def _depth_callback(self, msg: CompressedImage):
        try:
            depth = decode_compressed_depth_image(msg)
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
            with self._lock:
                # print("[REALSENSE SUB depth] lock aquired")
                self.latest_depth = depth
                # print("[REALSENSE SUB depth] lock released")

        except Exception as e:
            print(f"Error processing Depth image for {self.camera_name}: {e}")
            # Set a default depth image on error
            with self._lock:
                # print("[REALSENSE SUB depth] lock aquired")
                self.latest_depth = np.zeros((100, 100))
                # print("[REALSENSE SUB depth] lock released")


    def _info_callback(self, msg: CameraInfo):
        with self._lock:
            # print("[REALSENSE SUB info] lock aquired")
            self.latest_info = msg
            # print("[REALSENSE SUB info] lock released")

        source_frame = msg.header.frame_id
        #print(f"Received CameraInfo for {self.camera_name_space}: {source_frame}")
        time.sleep(0.01)  # Give some time for TF to be available
        pose = self.tf_handler.lookup_transform(config["base_frame"],source_frame, msg.header.stamp)

        with self._lock:
            # print("[REALSENSE SUB info] lock aquired")
            self.latest_pose = pose
            # print("[REALSENSE SUB info] lock released")


    def get_data(self):
        rgb, depth, info, pose = None, None, None, None
        if self.latest_rgb is not None and self.latest_depth is not None and self.latest_info is not None:
            with self._lock:
                # print("[REALSENSE SUB get data] lock aquired")

                rgb = self.latest_rgb
                depth = self.latest_depth
                info = self.latest_info
                pose = self.latest_pose

                # Clear buffers
                self.latest_rgb = None
                self.latest_depth = None
                self.latest_info = None
                self.latest_pose = None
                # print("[REALSENSE SUB get data] lock released")

        return rgb, depth, info, pose

    def shutdown(self):
        # Stop spinning and clean up
        self._executor.shutdown()
        self._spin_thread.join(timeout=1.0)
        self.destroy_node()

    def __str__(self):
        return f"RealSenseSubscriber(camera_name={self.camera_name})"

    def __repr__(self):
        return self.__str__()


def TestSubscriber(args=None):
    """Example usage of RealSenseSubscriber."""
    rclpy.init(args=args)
    print(f"hello world")
    name_spaces = config["rs_name_spaces"]
    subs = [RealSenseSubscriber(name_space) for name_space in name_spaces]
    cam_dir = os.path.join(fig_dir, 'realsense_images')
    os.makedirs(cam_dir, exist_ok=True)
    i = 0
    # Create OpenCV windows
    for name_space in name_spaces:
        cv2.namedWindow(f"{name_space}/RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"{name_space}/Depth", cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            for sub in subs:
                rgb, depth, info, pose = sub.get_data()
                if info is not None and rgb is not None and depth is not None:
                    # Display pose info
                    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    pose_text = f"Pose: {pose}" if pose else "Pose: None"
                    cv2.putText(rgb, pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow(f"{sub.camera_name_space}/RGB", rgb)
                    cv2.imshow(f"{sub.camera_name_space}/Depth", depth)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                save_dir = os.path.join(cam_dir, f"set_{i}")
                os.makedirs(save_dir, exist_ok=True)
                i += 1
                for sub in subs:
                    rgb = None
                    while rgb is None:
                        rgb, depth, info, pose = sub.get_data()
                    name = sub.camera_name_space.replace('/', '_')
                    rgb_path = os.path.join(save_dir, f"{name}_rgb.png")
                    depth_path = os.path.join(save_dir, f"{name}_depth.png")
                    info_path = os.path.join(save_dir, f"{name}_info.txt")
                    pose_path = os.path.join(save_dir, f"{name}_pose.txt")
                    cv2.imwrite(depth_path, depth)
                    cv2.imwrite(rgb_path, rgb)
                    with open(info_path, 'w') as f:
                        f.write(str(info))
                    with open(pose_path, 'w') as f:
                        f.write(str(pose))
                    print(f"Saved RGB image to {rgb_path}")


    except KeyboardInterrupt:
        pass
    finally:
        for sub in subs:
            sub.shutdown()
        rclpy.shutdown()
        cv2.destroyAllWindows()


def TestFoundationModels(args=None):
    from BBBackBones import Gemini_BB, OWLv2, YOLO_WORLD 
    from SAM2 import SAM2_PC
    from visualization_msgs.msg import Marker, MarkerArray
    from ros_utils import pcd_to_msg, box_to_marker, text_marker
    import random
    rclpy.init(args=args)
    sub = RealSenseSubscriber(config["rs_name_spaces"][0])
    # QoS for sensor data (Best effort + TRANSIENT_LOCAL)
    sensor_data_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5
    )

    pc_pub = sub.create_publisher(PointCloud2, '/foundation_models/pointclouds', sensor_data_qos)
    marker_pub = sub.create_publisher(MarkerArray, '/foundation_models/markers', sensor_data_qos)
    img_pub = sub.create_publisher(Image, '/foundation_models/predictions_2d', sensor_data_qos)
    #BB = Gemini_BB()
    #BB = OWLv2()
    BB = YOLO_WORLD()
    SAM = SAM2_PC()
    do_debug = False
    queries = config["test_querys"]

    while rclpy.ok():
        rgb_img, depth_img, info, pose = None, None, None, None
        while rgb_img is None or depth_img is None or info is None or pose is None:    
            rgb_img, depth_img, info, pose = sub.get_data()
            if rgb_img is None or depth_img is None or info is None or pose is None:
                time.sleep(0.1)  # Wait before trying again
                continue

        #print(f"{info=}")
        intrinsics = {
            "fx": info.k[0],
            "fy": info.k[4],
            "cx": info.k[2],
            "cy": info.k[5],
            "width": info.width,
            "height": info.height
        }

        predictions_2d = BB.predict(rgb_img, queries, debug=do_debug)
        



        candidates_3d = {}
        for query_object, canditates in predictions_2d.items():
            #print("\n\n")
            point_clouds, boxes, probs, rgb_masks, depth_masks = SAM.predict(rgb_img, depth_img, canditates["boxes"], canditates["probs"], intrinsics, pose, debug=do_debug, query_str=query_object)
            candidates_3d[query_object] = {
                "pcds": point_clouds,
                "boxes": boxes,
                "probs": probs,
                "masked_rgb": rgb_masks,
                "masked_depth": depth_masks
            }
        clear_arr = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        clear_marker.header.stamp = sub.get_clock().now().to_msg()
        clear_marker.header.frame_id = config["base_frame"]

        clear_arr.markers.append(clear_marker)
        marker_pub.publish(clear_arr)


        Marker_Arr = MarkerArray()
        marker_id = 1
        pcd_acc = None
        box_counter = 0
        text_counter = 0
        pcd_counter = 0
        for q in candidates_3d:
            pcds = candidates_3d[q]["pcds"]
            for pcd in pcds:
                if pcd_acc is None:
                    pcd_acc = pcd
                else:    
                    pcd_acc = pcd_acc.append(pcd)
                pcd_counter += 1
            boxes = candidates_3d[q]["boxes"]
            probs = candidates_3d[q]["probs"]
            assert len(boxes) == len(probs)
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                box_counter += 1
                bbox_3d = box_to_marker(box.to_legacy(), [1-prob, prob, 0.0, 1.0], config["base_frame"], marker_id)
                Marker_Arr.markers.append(bbox_3d)
                marker_id += 1

                text_counter += 1
                prob_text = text_marker(f"{q.replace(' ', '')}_{i}:{prob:.2f}", box.to_legacy().get_center().tolist(), [1-prob, prob, 0.0, 1.0], config["base_frame"], marker_id)
                Marker_Arr.markers.append(prob_text)
                marker_id += 1
                
               
        if pcd_acc is not None:
            pc_pub.publish(pcd_to_msg(pcd_acc, config['base_frame']))
        marker_pub.publish(Marker_Arr)
        annoted_image = display_2dCandidates(rgb_img, predictions_2d, display=False, save_path=None)
        img_msg = sub.bridge.cv2_to_imgmsg(annoted_image, encoding='bgr8')
        img_msg.header.stamp = sub.get_clock().now().to_msg()
        img_msg.header.frame_id = info.header.frame_id
        img_pub.publish(img_msg)
        print(f"published Pointclouds: {pcd_counter}, Boxes:{box_counter}, Text: {text_counter}")
    return None