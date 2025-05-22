#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from realsense2_camera_msgs.msg import RGBD
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class RealSenseSubscriber(Node):
    """
    Subscribes to RGB, Depth, and RGBD topics for a given camera namespace under /realsense.
    Displays each stream in its own OpenCV window.

    camera_key: e.g. 'head' or 'left_hand'.
    display_rate_sec: how often (seconds) to refresh the windows.
    """
    def __init__(self, camera_key: str, display_rate_sec: float = 1.0):
        ns_key = camera_key.replace(' ', '_')
        node_name = f'{ns_key}_subscriber'
        super().__init__(node_name)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        print(f"[INIT] Node name: {node_name}")

        self.ns = f'realsense/{ns_key}'  # no leading slash for window names
        print(f"[INIT] Subscriber namespace: /{self.ns}")

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.rgbd_image = None

        # Create separate windows
        self.win_rgb = f'{self.ns}/color'
        self.win_depth = f'{self.ns}/aligned_depth'
        self.win_rgbd_rgb = f'{self.ns}/rgbd_rgb'
        self.win_rgbd_depth = f'{self.ns}/rgbd_depth'
        for w in [self.win_rgb, self.win_depth, self.win_rgbd_rgb, self.win_rgbd_depth]:
            cv2.namedWindow(w, cv2.WINDOW_NORMAL)
            print(f"[INIT] Created window: {w}")

        # Subscriptions
        topic_rgb = f'/{self.ns}/color/image_raw'
        self.create_subscription(Image, topic_rgb, self._rgb_callback, sensor_qos)
        print(f"[INIT] Subscribed to RGB topic: {topic_rgb}")

        topic_depth = f'/{self.ns}/aligned_depth_to_color/image_raw'
        self.create_subscription(Image, topic_depth, self._depth_callback, sensor_qos)
        print(f"[INIT] Subscribed to Depth topic: {topic_depth}")

        topic_rgbd = f'/{self.ns}/rgbd'
        self.create_subscription(RGBD, topic_rgbd, self._rgbd_callback, sensor_qos)
        print(f"[INIT] Subscribed to RGBD topic: {topic_rgbd}")

        # Timer for display
        self.create_timer(display_rate_sec, self.display_images)
        print(f"[INIT] Display timer set to {display_rate_sec} sec")

    def _rgb_callback(self, msg: Image):
        self.rgb_image = msg
        print(f"[CALLBACK] RGB received @ {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

    def _depth_callback(self, msg: Image):
        self.depth_image = msg
        print(f"[CALLBACK] Depth received @ {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

    def _rgbd_callback(self, msg: RGBD):
        self.rgbd_image = msg
        print(f"[CALLBACK] RGBD received @ {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

    def display_images(self):
        print("[DISPLAY] Refreshing windows")

        # RGB
        if self.rgb_image:
            cv_rgb = self.bridge.imgmsg_to_cv2(self.rgb_image, 'bgr8')
        else:
            cv_rgb = np.zeros((480,640,3), np.uint8)
        cv2.imshow(self.win_rgb, cv_rgb)

        # Depth
        if self.depth_image:
            d_raw = self.bridge.imgmsg_to_cv2(self.depth_image, 'passthrough')
            d_norm = cv2.normalize(d_raw, None, 0, 255, cv2.NORM_MINMAX)
            cv_depth = cv2.cvtColor(d_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            cv_depth = np.zeros_like(cv_rgb)
        cv2.imshow(self.win_depth, cv_depth)

        # RGBD rgb
        if self.rgbd_image:
            cv_rgbd_rgb = self.bridge.imgmsg_to_cv2(self.rgbd_image.rgb, 'bgr8')
        else:
            cv_rgbd_rgb = np.zeros_like(cv_rgb)
        cv2.imshow(self.win_rgbd_rgb, cv_rgbd_rgb)

        # RGBD depth
        if self.rgbd_image:
            dr = self.bridge.imgmsg_to_cv2(self.rgbd_image.depth, 'passthrough')
            dr_norm = cv2.normalize(dr, None, 0, 255, cv2.NORM_MINMAX)
            cv_rgbd_depth = cv2.cvtColor(dr_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            cv_rgbd_depth = np.zeros_like(cv_rgb)
        cv2.imshow(self.win_rgbd_depth, cv_rgbd_depth)

        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    camera_key = sys.argv[1] if len(sys.argv)>1 else 'left_hand'
    print(f"[MAIN] Starting subscriber for camera: {camera_key}")
    node = RealSenseSubscriber(camera_key, display_rate_sec=1.0)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("[MAIN] Shutdown request")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()