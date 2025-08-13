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
if ros_dir not in sys.path:
    sys.path.insert(0, ros_dir)
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)


config_path = os.path.join(parent_dir, 'config.json')
config = json.load(open(config_path, 'r'))


#from SAM2 import SAM2_PC
#from BBBackBones import Gemini_BB, OWLv2
from ros_utils import decode_compressed_depth_image, TFHandler
from math_utils import pose_to_matrix, matrix_to_pose


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
            depth=5
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
            f"{camera_name_space}/depth/camera_info",
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
                self.latest_rgb = rgb_img
        except Exception as e:
            print(f"Error processing RGB image for {self.camera_name}: {e}")

    def _depth_callback(self, msg: CompressedImage):
        try:
            depth = decode_compressed_depth_image(msg)
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
            with self._lock:
                self.latest_depth = depth
        except Exception as e:
            print(f"Error processing Depth image for {self.camera_name}: {e}")
            # Set a default depth image on error
            with self._lock:
                self.latest_depth = np.zeros((100, 100))

    def _info_callback(self, msg: CameraInfo):
        with self._lock:
            self.latest_info = msg
        source_frame = msg.header.frame_id
        #print(f"Received CameraInfo for {self.camera_name_space}: {source_frame}")
        pose = self.tf_handler.lookup_pose(config["base_frame"],source_frame, msg.header.stamp)

        with self._lock:
            self.latest_pose = pose

    def get_data(self):
        rgb, depth, info, pose = None, None, None, None
        if self.latest_rgb is not None and self.latest_depth is not None and self.latest_info is not None:
            with self._lock:
                rgb = self.latest_rgb
                depth = self.latest_depth
                info = self.latest_info
                pose = self.latest_pose

                # Clear buffers
                self.latest_rgb = None
                self.latest_depth = None
                self.latest_info = None
                self.latest_pose = None
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


def annotate_detections(img_np: np.ndarray, detections: dict, score_thresh: float = 0.0) -> np.ndarray:
    """
    Returns a copy of the image with boxes and labels drawn.
    Box & text color: Red = 1 - prob, Green = prob.
    
    Parameters
    ----------
    img_np : np.ndarray
        HxWx3 uint8 NumPy image (BGR format expected for OpenCV drawing).
    detections : dict
        {
            "label1": {"boxes": [[x1,y1,x2,y2], ...], "probs": [p1, p2, ...]},
            ...
        }
    score_thresh : float
        Minimum probability for drawing a detection.
    
    Returns
    -------
    np.ndarray
        Annotated image (BGR).
    """
    
    def prob_to_color(prob: float) -> tuple:
        """Convert probability to BGR color."""
        r = int((1.0 - prob) * 255)
        g = int(prob * 255)
        return (0, g, r)  # OpenCV uses BGR

    def draw_box(img, box, label_text, color, thickness=2):
        """Draw rectangle and filled label background."""
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(img.shape[1], img.shape[0]) / 1000.0)
        t          = max(1, thickness)
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, t)

        tb_x1, tb_y1 = x1, max(0, y1 - th - baseline - 3)
        tb_x2, tb_y2 = x1 + tw + 6, tb_y1 + th + baseline + 3
        cv2.rectangle(img, (tb_x1, tb_y1), (tb_x2, tb_y2), color, -1)
        cv2.putText(img, label_text, (tb_x1 + 3, tb_y2 - baseline - 2),
                    font, font_scale, (0, 0, 0), t, cv2.LINE_AA)

    # Work on a copy
    annotated_img = img_np.copy()

    for label, data in detections.items():
        boxes = data.get("boxes", [])
        probs = data.get("probs", [])
        n = min(len(boxes), len(probs)) if probs else len(boxes)

        for i in range(n):
            p = probs[i] if probs and i < len(probs) else 0.0
            if p < score_thresh:
                continue
            color = prob_to_color(p)
            text = f"{label} {p:.2f}"
            draw_box(annotated_img, boxes[i], text, color)

    return annotated_img

def TestFoundationModels(args=None):
    from BBBackBones import Gemini_BB, OWLv2, YOLO_WORLD 
    from SAM2 import SAM2_PC
    from visualization_msgs.msg import Marker, MarkerArray
    from ros_utils import pcd_to_msg, box_to_marker, text_marker
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
        
        Marker_Arr = MarkerArray()
        marker_id = 1
        pcd_acc = None
        for q in candidates_3d:
            for pcd in candidates_3d[q]["pcds"]:
                if pcd_acc is None:
                    pcd_acc = pcd
                else:    
                    pcd_acc.append(pcd)
            boxes = candidates_3d[q]["boxes"]
            probs = candidates_3d[q]["probs"]
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                bbox_3d = box_to_marker(box.to_legacy(), [1-prob, prob, 0.0, 1.0], config["base_frame"], marker_id)
                Marker_Arr.markers.append(bbox_3d)
                marker_id += 1
                prob_text = text_marker(f"{q}_{i}: {prob:.2f}", box.get_center().numpy().tolist(), [1-prob, prob, 0.0, 1.0], config["base_frame"], marker_id)
                Marker_Arr.markers.append(prob_text)
                marker_id += 1
        
        pc_pub.publish(pcd_to_msg(pcd_acc, config['base_frame']))
        marker_pub.publish(Marker_Arr)
        annoted_image = annotate_detections(rgb_img, predictions_2d, score_thresh=0.0)
        img_msg = sub.bridge.cv2_to_imgmsg(annoted_image, encoding='bgr8')
        img_msg.header.stamp = sub.get_clock().now().to_msg()
        img_msg.header.frame_id = info.header.frame_id
        img_pub.publish(img_msg)
        print("published")
    return None