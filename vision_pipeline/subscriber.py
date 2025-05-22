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

class RealSenseSubscriber(Node):
    """
    Subscribes to the RGBD and CameraInfo topics for a given camera namespace under /realsense.
    Stores the latest RGB-D images for display.

    camera_key: e.g. 'head' or 'left_hand'.
    """
    def __init__(self, camera_key: str):
        node_name = f"{camera_key.replace(' ', '_')}_subscriber"
        super().__init__(node_name)
        self.camera_key = camera_key
        self.bridge = CvBridge()
        self.latest_rgb = None      # Latest BGR image
        self.latest_depth = None    # Latest depth image (normalized for display)
        self._lock = threading.Lock()

        # QoS for sensor data
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscribe to combined RGBD topic
        self.create_subscription(
            RGBD,
            f"/realsense/{camera_key}/rgbd",
            self._rgbd_callback,
            qos
        )

        # Spin in background thread
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True
        )
        self._spin_thread.start()

    def _rgbd_callback(self, msg: RGBD):
        # Convert and store latest RGB and depth images
        rgb_img = self.bridge.imgmsg_to_cv2(msg.rgb, 'bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(msg.depth, 'passthrough')
        # Normalize depth for display
        depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = cv2.cvtColor(depth_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        with self._lock:
            self.latest_rgb = rgb_img
            self.latest_depth = depth_display

    def display(self):
        """
        Display the latest RGB and depth images in OpenCV windows.
        """
        with self._lock:
            rgb = self.latest_rgb.copy() if self.latest_rgb is not None else np.zeros((480, 640, 3), np.uint8)
            depth = self.latest_depth.copy() if self.latest_depth is not None else np.zeros((480, 640, 3), np.uint8)

        cv2.imshow(f"{self.camera_key}/RGB", rgb)
        cv2.imshow(f"{self.camera_key}/Depth", depth)

    def shutdown(self):
        """
        Stop internal executor and destroy node.
        """
        self._executor.shutdown()
        self.destroy_node()


def main(args=None):
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
                sub.display()

            # WaitKey for refresh and exit
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        for sub in subs:
            sub.shutdown()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
