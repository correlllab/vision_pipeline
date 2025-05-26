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

from vision_pipeline.VisionPipe import VisionPipe
from vision_pipeline.foundation_models import OWLv2, SAM2_PC
import cv2

class RealSenseSubscriber(Node):
    """
    Subscribes to the RGBD and CameraInfo topics for a given camera namespace under /realsense.
    Stores the latest RGB-D images and camera info for display or further processing.

    camera_key: e.g. 'head' or 'left_hand'.
    """
    def __init__(self, camera_key: str):
        node_name = f"{camera_key.replace(' ', '_')}_subscriber"
        super().__init__(node_name)
        self.camera_key = camera_key
        self.bridge = CvBridge()
        self.latest_rgb = None      # Latest BGR image
        self.latest_depth = None    # Latest depth image (normalized for display)
        self.latest_info = None     # Latest CameraInfo
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
        # Subscribe to camera info for intrinsics
        self.create_subscription(
            CameraInfo,
            f"/realsense/{camera_key}/color/camera_info",
            self._info_callback,
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
        #depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        #depth_display = cv2.cvtColor(depth_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        with self._lock:
            self.latest_rgb = rgb_img
            self.latest_depth = depth_img

    def _info_callback(self, msg: CameraInfo):
        # Store the latest camera info message
        with self._lock:
            self.latest_info = msg
    
    def get_intrinsics(self):
        """
        Returns the color camera intrinsics from the latest CameraInfo:
        {fx, fy, cx, cy, width, height, model, coeffs}
        """
        if self.latest_info is None:
            return None

        ci = self.latest_info
        # K is row-major [k00, k01, k02; k10, k11, k12; k20, k21, k22]
        fx = ci.k[0]
        fy = ci.k[4]
        cx = ci.k[2]
        cy = ci.k[5]

        return {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'width':  ci.width,
            'height': ci.height,
            'model':  ci.distortion_model,
            'coeffs': list(ci.d)  # usually [k1, k2, t1, t2, k3]
        }

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


class ROS_VisionPipe(VisionPipe):
    def __init__(self):
        super().__init__()
        self.sub = RealSenseSubscriber("head")
        self.track_strings = []

    def update(self, debug=False):
        rgb_img = self.sub.latest_rgb
        depth_img = self.sub.latest_depth
        intrinsics = self.sub.get_intrinsics()
        pose = [0,0,0,0,0,0]
        if rgb_img is None or depth_img is None or intrinsics is None:
            print("No image received yet.")
            return False
        if len(self.track_strings) == 0:
            print("No track strings provided.")
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


def TestFoundationModels(args=None):
    rclpy.init(args=args)
    sub = RealSenseSubscriber("head")
    OWL = OWLv2()
    SAM = SAM2_PC()
    while True:
        rgb_img = sub.latest_rgb
        depth_img = sub.latest_depth
        print("RGB img shape: ", rgb_img.shape)
        print("Depth img shape: ", depth_img.shape)
        intrinsics = sub.get_intrinsics()
        obs_pose = [0, 0, 0, 0, 0, 0]
        if rgb_img is None or depth_img is None or intrinsics is None:
            print("Waiting for images...")
            continue
        print("RGB img shape: ", rgb_img.shape)
        print("Depth img shape: ", depth_img.shape)
        querries = ["drill", "screw driver", "wrench"]
        predictions_2d = OWL.predict(rgb_img, querries, debug=True)
        for querry_object, canditates in predictions_2d.items():
            print("\n\n")
            point_clouds, boxes, scores = SAM.predict(rgb_img, depth_img, canditates["boxes"], canditates["scores"], intrinsics, debug=True)


def TestVisionPipe(args=None):
    rclpy.init(args=args)
    VP = ROS_VisionPipe()
    VP.add_track_string("drill")
    VP.add_track_string(["wrench", "screwdriver"])
    success_counter = 0
    while success_counter < 5:
        success = VP.update()
        success_counter += 1 if success != False else 0

    print("Success counter: ", success_counter)

    for object, predictions in VP.tracked_objects.items():
        print(f"{object=}")
        print(f"   {len(predictions['boxes'])=}, {len(predictions['pcds'])=}, {predictions['scores'].shape=}")
        for i, pcd in enumerate(predictions["pcds"]):
            print(f"   {i=}, {predictions['scores'][i]=}")
    print("")

    VP.display()