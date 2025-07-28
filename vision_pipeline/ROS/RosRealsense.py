#!/usr/bin/env python3
import sys
import threading
from cv_bridge import CvBridge
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2


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
from ros_utils import decode_compressed_depth_image, msg_to_pcd, pcd_to_msg, TFHandler, transform_to_matrix
from math_utils import pose_to_matrix, matrix_to_pose


class RealSenseSubscriber(Node):
    def __init__(self, camera_name):
        print("Initializing RealSenseSubscriber for camera:", camera_name)
        node_name = f"{camera_name.replace(' ', '_')}_subscriber"
        super().__init__(node_name)
        self.camera_name = camera_name
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_info = None
        self.latest_pose = None
        self._lock = threading.Lock()
        cam_idx = -1
        try:
            cam_idx = config["rs_names"].index(camera_name)
        except ValueError:
            self.get_logger().error(f"Camera name {camera_name} not found in config.")
            raise ValueError(f"Camera name {camera_name} not found in config {config['rs_names']=}.")
        self.source_frame = config["rs_frames"][cam_idx]
        # TF2 buffer and listener with longer cache time
        self.tf_handler = TFHandler(self)

        # QoS for image topics (RELIABLE + TRANSIENT_LOCAL)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # QoS for camera_info (RELIABLE + VOLATILE)
        info_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscriptions
        self.create_subscription(
            CompressedImage,
            f"/realsense/{camera_name}/color/image_raw/compressed",
            self._rgb_callback,
            image_qos,
        )
        self.create_subscription(
            CompressedImage,
            f"/realsense/{camera_name}/aligned_depth_to_color/image_raw/compressedDepth",
            self._depth_callback,
            image_qos,
        )
        self.create_subscription(
            CameraInfo,
            f"/realsense/{camera_name}/color/camera_info",
            self._info_callback,
            info_qos,
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

        pose = self.tf_handler.lookup_pose(config["base_frame"], self.source_frame, msg.header.stamp)

        with self._lock:
            self.latest_pose = pose

    def get_data(self):
        rgb, depth, info, pose = None, None, None, None
        if self.latest_rgb is not None and self.latest_depth is not None and self.latest_info is not None:
            with self._lock:
                rgb = self.latest_rgb.copy()  # Make copies to avoid race conditions
                depth = self.latest_depth.copy()
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
    cams = config["rs_names"]
    subs = [RealSenseSubscriber(cam) for cam in cams]
    cam_dir = os.path.join(fig_dir, 'realsense_images')
    os.makedirs(cam_dir, exist_ok=True)
    i = 0
    # Create OpenCV windows
    for cam in cams:
        cv2.namedWindow(f"{cam}/RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"{cam}/Depth", cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            for sub in subs:
                rgb, depth, info, pose = sub.get_data()
                if info is not None and rgb is not None and depth is not None:
                    # Display pose info
                    pose_text = f"Pose: {pose}" if pose else "Pose: None"
                    cv2.putText(rgb, pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow(f"{sub.camera_name}/RGB", rgb)
                    cv2.imshow(f"{sub.camera_name}/Depth", depth)

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
                    rgb_path = os.path.join(save_dir, f"{sub.camera_name}_rgb.png")
                    cv2.imwrite(rgb_path, rgb)
                    print(f"Saved RGB image to {rgb_path}")


    except KeyboardInterrupt:
        pass
    finally:
        for sub in subs:
            sub.shutdown()
        rclpy.shutdown()
        cv2.destroyAllWindows()


class PointCloudAccumulator(Node):
    def __init__(self, camera_names):
        node_name = "point_cloud_accumulator"
        super().__init__(node_name)
        pc_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.subscribers = []
        for camera_name in camera_names:
            topic = f"/realsense/{camera_name}/depth/color/points"
            sub = self.create_subscription(
                PointCloud2,
                topic,
                self.pc_callback,
                pc_qos,
            )
            print(f"Subscribed to point cloud for {topic}")
            self.subscribers.append(sub)

        self.publisher = self.create_publisher(
            PointCloud2,
            "/realsense/accumulated_point_cloud",
            pc_qos
        )
        self.pcd = o3d.geometry.PointCloud()
        self.tf_handler = TFHandler(self)
        self._lock = threading.Lock()
        self.msg_queue = []

        # Start a timer to publish the accumulated point cloud at 6 Hz
        self._timer = self.create_timer(1.0 / 6.0, self.publish_accumulated_pc)

        # Private executor & spin thread for this node
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._spin_thread.start()


    def publish_accumulated_pc(self):
        if len(self.pcd.points) == 0:
            # print("No point cloud data to publish.")
            # # Sample points from a unit sphere and add to the point cloud for visualization/testing
            # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
            # sphere_pcd = sphere.sample_points_uniformly(number_of_points=500)
            # sphere_pcd.paint_uniform_color([1, 0, 0])  # Red color for visibility
            # self.pcd += sphere_pcd
            return
        # Publish accumulated point cloud
        msg_out = pcd_to_msg(self.pcd, frame_id=config["base_frame"])
        self.publisher.publish(msg_out)
        # print(f"Published accumulated point cloud with {len(self.pcd.points)} points to /realsense/accumulated_point_cloud")

    def pc_callback(self, msg):
        rs_frame = msg.header.frame_id
        matching_frames = [f for f in config["rs_names"] if f in rs_frame][0]
        cam_idx = config["rs_names"].index(matching_frames)
        if cam_idx < 0:
            # print(f"Camera frame {frame} matching {matching_frames} not found in config, skipping point cloud.")
            return
        frame = config["rs_frames"][cam_idx]
        # transform_ros = self.tf_handler.lookup_transform(config["base_frame"], frame, msg.header.stamp)
        # if transform_ros is None:
        #     print(f"Transform not found for frame {frame}->{config['base_frame']}, skipping point cloud.")
        #     continue
        pose = self.tf_handler.lookup_pose(config["base_frame"], frame, msg.header.stamp)
        if pose is None:
            # print(f"Pose not found for frame {frame}->{config['base_frame']}, skipping point cloud.")
            return
        with self._lock:
            # print(f"Pose found for frame {frame}->{config['base_frame']}.")
            
            self.msg_queue.append((msg, pose))

            # print(f"Received point cloud from {msg.header.frame_id}")
            
    def main_loop(self):
        while rclpy.ok():
            if len(self.msg_queue) > 0:
                print(f"\rProcessing {len(self.msg_queue)} point clouds in queue...", end="", flush=True)
                with self._lock:
                    msg, pose = self.msg_queue.pop(0)
                    #print(f"Processing point cloud from frame: {frame}")
                    if pose is None:
                        continue
                    transform = pose_to_matrix(pose)
                    #transform = transform_to_matrix(transform_ros)
                    pcd = msg_to_pcd(msg)
                    pcd.transform(transform)
                    if pcd.is_empty():
                        # print("Received empty point cloud, skipping...")
                        continue
                    self.pcd += pcd
                    self.pcd = self.pcd.voxel_down_sample(voxel_size=config["voxel_size"])
                    # Apply statistical outlier removal to denoise the point cloud
                    if config["statistical_outlier_removal"]:
                        self.pcd, ind = self.pcd.remove_statistical_outlier(nb_neighbors=config["statistical_nb_neighbors"], std_ratio=config["statistical_std_ratio"])
                    if config["radius_outlier_removal"]:
                        self.pcd, ind = self.pcd.remove_radius_outlier(nb_points=config["radius_nb_points"], radius=config["radius_radius"])

def TestPointCloudAccumulator(args=None):
    rclpy.init(args=args)
    camera_names = config["rs_names"]
    PC_ACC = PointCloudAccumulator(camera_names)
    PC_ACC.main_loop()
    rclpy.shutdown()

def TestFoundationModels(args=None):
    from BBBackBones import Gemini_BB, OWLv2, display_2dCandidates
    from SAM2 import SAM2_PC, display_3dCandidates
    rclpy.init(args=args)
    sub = RealSenseSubscriber("head")
    GEM = Gemini_BB()
    OWL = OWLv2()
    SAM = SAM2_PC()
    while rclpy.ok():
        rgb_img, depth_img, info, pose = sub.get_data()
        if rgb_img is None or depth_img is None or info is None:
            print("Waiting for images...")
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
        if rgb_img is None or depth_img is None or info is None:
            print("Waiting for images...")
            continue

        print("RGB img shape: ", rgb_img.shape)
        print("Depth img shape: ", depth_img.shape)
        queries = config["test_querys"]
        predictions_2d = OWL.predict(rgb_img, queries, debug=True)
        display_2dCandidates(rgb_img, predictions_2d, window_prefix="OWL_")
        predictions_2d = GEM.predict(rgb_img, queries, debug=True)
        display_2dCandidates(rgb_img, predictions_2d, window_prefix="GEM_")
        candidates_3d = {}
        for query_object, canditates in predictions_2d.items():
            #print("\n\n")
            point_clouds, boxes, probs,  rgb_masks, depth_masks = SAM.predict(rgb_img, depth_img, canditates["boxes"], canditates["probs"], intrinsics, debug=True, query_str=query_object)
            candidates_3d[query_object] = {
                "pcds": point_clouds,
                "boxes": boxes,
                "probs": probs,
                "masked_rgb": rgb_masks,
                "masked_depth": depth_masks
            }
        display_3dCandidates(candidates_3d, window_prefix="SAM2_")
    return None
