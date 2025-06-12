#!/usr/bin/env python3
import sys
import threading
import time
import open3d as o3d
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import CameraInfo, CompressedImage

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException
from tf2_ros         import LookupException, ConnectivityException, ExtrapolationException


import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)

from FoundationModels import OWLv2, SAM2_PC
from utils import quat_to_euler, decode_compressed_depth_image


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
        self.target_frame = "pelvis"

        # TF2 buffer and listener with longer cache time
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS for image topics (RELIABLE + TRANSIENT_LOCAL)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # QoS for camera_info (RELIABLE + VOLATILE)
        info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
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

        # Look up pose asynchronously - don't block the callback
        # Option 1: Use current time instead of message timestamp
        #pose = self.lookup_pose_current_time()

        # Option 2: If you need the exact timestamp, do it non-blockingly
        pose = self.lookup_pose_async(msg.header.stamp)

        with self._lock:
            self.latest_pose = pose

    def lookup_pose_current_time(self):
        """Look up the current pose (most recent available)"""
        source_frame = {
            "head": "head_camera_link",
            "left_hand": "left_hand_camera_link",
            "right_hand": "right_hand_camera_link",
        }.get(self.camera_name)

        if source_frame is None:
            self.get_logger().error(f"Unknown camera name {self.camera_name}")
            return None

        try:
            # Use Time(0) to get the most recent transform available
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time()  # This gets the latest available transform
            )

            # Build 6-DoF pose
            t = transform.transform.translation
            q = transform.transform.rotation
            roll, pitch, yaw = quat_to_euler(q.x, q.y, q.z, q.w)
            return [t.x, t.y, t.z, roll, pitch, yaw]

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed {source_frame}->{self.target_frame}: {e}")
            return None

    def lookup_pose_async(self, stamp_msg):
        """Alternative: Look up pose with specific timestamp but with timeout handling"""
        source_frame = {
            "head": "head_camera_link",
            "left_hand": "left_hand_camera_link",
            "right_hand": "right_hand_camera_link",
        }.get(self.camera_name)

        if source_frame is None:
            self.get_logger().error(f"Unknown camera name {self.camera_name}")
            return None

        stamp = Time.from_msg(stamp_msg)

        # Check if transform is available with a short timeout
        if not self.tf_buffer.can_transform(
            self.target_frame,
            source_frame,
            stamp,
            Duration(seconds=0.1)  # Very short timeout
        ):
            # Fall back to most recent transform
            self.get_logger().debug(f"Historical TF not available, using latest for {source_frame}")
            return self.lookup_pose_current_time()

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                stamp
            )

            # Build 6-DoF pose
            t = transform.transform.translation
            q = transform.transform.rotation
            roll, pitch, yaw = quat_to_euler(q.x, q.y, q.z, q.w)
            return [t.x, t.y, t.z, roll, pitch, yaw]

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed {source_frame}->{self.target_frame}: {e}")
            # Fall back to current time
            return self.lookup_pose_current_time()

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
    cams = ['head', 'left_hand', 'right_hand']
    subs = [RealSenseSubscriber(cam) for cam in cams]

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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        for sub in subs:
            sub.shutdown()
        rclpy.shutdown()
        cv2.destroyAllWindows()

def TestFoundationModels(args=None):
    rclpy.init(args=args)
    sub = RealSenseSubscriber("head")
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
        querries = ["drill", "screw driver", "wrench"]
        predictions_2d = OWL.predict(rgb_img, querries, debug=True)
        for query_object, canditates in predictions_2d.items():
            #print("\n\n")
            point_clouds, boxes, scores,  rgb_masks, depth_masks = SAM.predict(rgb_img, depth_img, canditates["boxes"], canditates["scores"], intrinsics, debug=True)
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
