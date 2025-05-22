#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from realsense2_camera_msgs.msg import RGBD
from cv_bridge import CvBridge
import matplotlib.pyplot as plt


class RealSenseSubscriber(Node):
    """
    Subscribes to RGB, Depth, and RGBD topics for a given camera namespace under /realsense.
    Use get_rgb(), get_depth(), get_rgbd() to retrieve the most recently
    received message for each stream. Call plot_images() to display them.

    camera_key: identifier for camera (e.g. 'head' or 'left hand').
    Spaces in camera_key are replaced with underscores to form a valid namespace.
    """
    def __init__(self, camera_key: str):
        # sanitize key for namespace
        ns_key = camera_key.replace(' ', '_')
        super().__init__(f'{ns_key}_subscriber')

        # base namespace for this camera
        self.ns = f'/realsense/{ns_key}'

        # cv_bridge for conversions
        self.bridge = CvBridge()

        # storage for latest messages
        self.rgb_image = None       # sensor_msgs.msg.Image
        self.depth_image = None     # sensor_msgs.msg.Image
        self.rgbd_image = None      # realsense2_camera_msgs.msg.RGBD

        # subscriptions
        self.create_subscription(
            Image,
            f'{self.ns}/color/image_raw',
            self._rgb_callback,
            10
        )
        self.create_subscription(
            Image,
            f'{self.ns}/aligned_depth_to_color/image_raw',
            self._depth_callback,
            10
        )
        self.create_subscription(
            RGBD,
            f'{self.ns}/rgbd',
            self._rgbd_callback,
            10
        )

    def _rgb_callback(self, msg: Image):
        self.rgb_image = msg

    def _depth_callback(self, msg: Image):
        self.depth_image = msg

    def _rgbd_callback(self, msg: RGBD):
        self.rgbd_image = msg

    def get_rgb(self) -> Image:
        """Return last received color Image (or None if none received yet)."""
        return self.rgb_image

    def get_depth(self) -> Image:
        """Return last received depth Image (or None if none received yet)."""
        return self.depth_image

    def get_rgbd(self) -> RGBD:
        """Return last received RGBD message (or None if none received yet)."""
        return self.rgbd_image

    def plot_images(self):
        """
        Plot RGB, Depth, and RGBD images using matplotlib.
        """
        # Prepare subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # RGB
        if self.rgb_image is not None:
            cv_rgb = self.bridge.imgmsg_to_cv2(self.rgb_image, desired_encoding='bgr8')
            axes[0].imshow(cv_rgb[..., ::-1])  # convert BGR→RGB
            axes[0].set_title('RGB')
            axes[0].axis('off')

        # Depth
        if self.depth_image is not None:
            cv_depth = self.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding='passthrough')
            axes[1].imshow(cv_depth, cmap='gray')
            axes[1].set_title('Depth')
            axes[1].axis('off')

        # RGBD (show RGB channel)
        if self.rgbd_image is not None:
            cv_rgbd_rgb = self.bridge.imgmsg_to_cv2(self.rgbd_image.rgb, desired_encoding='bgr8')
            axes[2].imshow(cv_rgbd_rgb[..., ::-1])
            axes[2].set_title('RGBD (RGB)')
            axes[2].axis('off')

        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    sub = RealSenseSubscriber('head')
    try:
        rclpy.spin(sub)
        while rclpy.ok():
            sub.plot_images()
            rclpy.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        sub.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
