#!/usr/bin/env python3
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


from sensor_msgs.msg import CameraInfo, Image
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
        self._lock = threading.Lock()

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
            Image,
            f"/realsense/{camera_name}/color/image_raw",
            self._rgb_callback,
            image_qos,
        )
        self.create_subscription(
            Image,
            f"/realsense/{camera_name}/aligned_depth_to_color/image_raw",
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

    def _rgb_callback(self, msg: Image):
        try:
            rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
            rgb_img = np.zeros((480, 640, 3), dtype=np.uint8)
        with self._lock:
            self.latest_rgb = rgb_img

    def _depth_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
            depth = np.zeros((480, 640), dtype=np.float32)
        with self._lock:
            self.latest_depth = depth

    def _info_callback(self, msg: CameraInfo):
        with self._lock:
            self.latest_info = msg

    def get_data(self):
        rgb, depth, info = None, None, None
        if self.latest_rgb is not None and self.latest_depth is not None and self.latest_info is not None:
            with self._lock:
                rgb = self.latest_rgb
                depth = self.latest_depth
                info = self.latest_info
                # clear buffers
                self.latest_rgb = None
                self.latest_depth = None
                self.latest_info = None
        return rgb, depth, info

    def shutdown(self):
        # stop spinning and clean up
        self._executor.shutdown()
        self._spin_thread.join(timeout=1.0)
        self.destroy_node()


def TestSubscriber(args=None):
    """Example usage of RealSenseSubscriber."""
    rclpy.init(args=args)
    cams = ['head', 'left_hand', 'right_hand']
    subs = [RealSenseSubscriber(cam) for cam in cams]

    # Create OpenCV windows
    for cam in cams:
        cv2.namedWindow(f"{cam}/RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"{cam}/Depth", cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            for sub in subs:
                rgb, depth, info = sub.get_data()
                if info is not None and rgb is not None and depth is not None:
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
        rgb_img, depth_img, info = sub.get_data()
        print(f"{info=}")
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
        obs_pose = [0, 0, 0, 0, 0, 0]
        
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


class ROS_VisionPipe(VisionPipe, Node):
    def __init__(self):
        super().__init__()
        self.sub = RealSenseSubscriber("head")
        self.track_strings = []

    def update(self, debug=False):
        rgb_img, depth_img, info = self.sub.get_data()
        pose = [0,0,0,0,0,0]
        if rgb_img is None or depth_img is None or info is None:
            print("No image received yet.")
            return False
        if len(self.track_strings) == 0:
            print("No track strings provided.")
            return False
        print(f"\n\n{info=}")
        intrinsics = {
            "fx": info.k[0],
            "fy": info.k[4],
            "cx": info.k[2],
            "cy": info.k[5],
            "width": info.width,
            "height": info.height
        }
        return super().update(rgb_img, depth_img, self.track_strings, intrinsics, pose, debug=debug)

    def add_track_string(self, new_track_string):
        if isinstance(new_track_string, str) and new_track_string not in self.track_strings:
            self.track_strings.append(new_track_string)
        elif isinstance(new_track_string, list):
            [self.track_strings.append(x) for x in new_track_string if x not in self.track_strings]
        else:
            raise ValueError("track_string must be a string or a list of strings")
    def remove_track_string(self, track_string):
        if track_string in self.track_strings:
            self.track_strings.remove(track_string)
        else:
            print(f"Track string '{track_string}' not found in tracked objects.")
            
def TestVisionPipe(args=None):
    rclpy.init(args=args)
    VP = ROS_VisionPipe()
    VP.add_track_string("drill")
    #VP.add_track_string(["wrench", "screwdriver"])
    success_counter = 0
    while success_counter < 5:
        success = VP.update()
        success_counter += 1 if success != False else 0
        if success:
            print(f"Success {success_counter} with {len(VP.tracked_objects)} tracked objects")
            VP.vis_belief2D(query="drill", blocking=False, prefix = f"T={success_counter} ")

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