#!/usr/bin/env python3
import sys
import threading
import time
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from cv_bridge import CvBridge

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point
from tf2_ros import Buffer, TransformListener, LookupException, TimeoutException, ConnectivityException
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time

from rclpy.duration import Duration

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2


from rclpy.time      import Time
from rclpy.duration  import Duration
from tf2_ros         import LookupException, ConnectivityException, ExtrapolationException


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


        # TF2 buffer and listener
        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
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
        #print(f"Received depth image for {self.camera_name}")
        try:
            rgb_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self._lock:
                self.latest_rgb = rgb_img
            #print(f"{self.camera_name} Received rgb image with shape: {rgb_img.shape}")
            
        except Exception as e:
            print(f"Error processing RGB image for {self.camera_name}: {e}")
            pass
        

    def _depth_callback(self, msg: CompressedImage):
        #print(f"Received depth image for {self.camera_name}")
        self.latest_depth = np.zeros((100,100))
        try:
            depth = decode_compressed_depth_image(msg)
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
            with self._lock:
                self.latest_depth = depth
            #print(f"{self.camera_name} Received depth image with shape: {depth.shape}")
        except Exception as e:
            print(f"Error processing Depth image for {self.camera_name}: {e}")
            pass
        

    def _info_callback(self, msg: CameraInfo):
        with self._lock:
            self.latest_info = msg
            self.latest_pose = self.lookup_pose(msg.header.stamp)
    
    def lookup_pose(self, stamp_msg):
        source_frame = {
            "head":       "head_camera_link",
            "left_hand":  "left_hand_camera_link",
            "right_hand": "right_hand_camera_link",
        }.get(self.camera_name)
        if source_frame is None:
            self.get_logger().error(f"Unknown camera name {self.camera_name}")
            return None

        # 1. Convert to rclpy.Time (tf2 prefers that type)
        stamp = Time.from_msg(stamp_msg)

        # 2. Wait until TF for *this* stamp is present
        if not self.tf_buffer.can_transform(self.target_frame,
                                            source_frame,
                                            stamp,
                                            Duration(seconds=2)):
            self.get_logger().warn(f"TF not available for {source_frame}->{self.target_frame} at {stamp.to_msg()}")
            return None            # Try again on the next CameraInfo

        try:
            transform = self.tf_buffer.lookup_transform(
                            self.target_frame,
                            source_frame,
                            stamp)
        except (LookupException,
                ConnectivityException,
                ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed {source_frame}->{self.target_frame}: {e}")
            return None

        # --- build 6-DoF pose -----------------------------------------
        t = transform.transform.translation
        q = transform.transform.rotation
        roll, pitch, yaw = quat_to_euler(q.x, q.y, q.z, q.w)
        return [t.x, t.y, t.z, roll, pitch, yaw]

    def get_data(self):
        rgb, depth, info, pose = None, None, None, None
        if self.latest_rgb is not None and self.latest_depth is not None and self.latest_info is not None:
            with self._lock:
                rgb = self.latest_rgb
                depth = self.latest_depth
                info = self.latest_info
                pose = self.latest_pose
                #print(f"{self.camera_name} - RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
                # clear buffers
                self.latest_rgb = None
                self.latest_depth = None
                self.latest_info = None
                self.latest_pose = None
        return rgb, depth, info, pose

    def shutdown(self):
        # stop spinning and clean up
        self._executor.shutdown()
        self._spin_thread.join(timeout=1.0)
        self.destroy_node()


def TestSubscriber(args=None):
    """Example usage of RealSenseSubscriber."""
    rclpy.init(args=args)
    print(f"hello world")
    cams =['head', 'left_hand', 'right_hand'] #['head', 'left_hand', 'right_hand']
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
                    
                    cv2.putText(rgb, f"{pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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
        if rgb_img is None or depth_img is None or info is None:
            print("Waiting for images...")
            continue
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


def box_to_points(box):
    # min and max
    min_pt = np.array(box.min_bound)
    max_pt = np.array(box.max_bound)

    # 8 corners
    corners = [
        [min_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]]
    ]

    # 12 edges as pairs of indices into corners list
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]

    # Build list of Points for LINE_LIST (pairs form lines)
    points = []
    for i, j in edges:
        p1 = Point(x=corners[i][0], y=corners[i][1], z=corners[i][2])
        p2 = Point(x=corners[j][0], y=corners[j][1], z=corners[j][2])
        points.append(p1)
        points.append(p2)
    return points




class ROS_VisionPipe(VisionPipe, Node):
    def __init__(self):
        VisionPipe.__init__(self)
        Node.__init__(self, "ros_vision_pipe")
        self.sub = RealSenseSubscriber("head")
        self.track_strings = []
        self.marker_publishers = {}
        self.pc_publishers = {}
        self._lock = threading.Lock()
        self._start_publishers()

    def _start_publishers(self, rate_hz=10):
        with self._lock:
            self.new_data = True
        self._pub_thread = threading.Thread(target=self._publish_loop, args=(rate_hz,))
        self._pub_thread.start()

    def add_track_string(self, new_track_string):
        if isinstance(new_track_string, str) and new_track_string not in self.track_strings:
            self.track_strings.append(new_track_string)
            topic_sub_name = new_track_string.replace(" ", "_")
            self.marker_publishers[new_track_string] = self.create_publisher(MarkerArray, f"/tracked_objects/markers/{topic_sub_name}", 1)
            self.pc_publishers[new_track_string] = self.create_publisher(PointCloud2, f"/tracked_objects/pointcloud/{topic_sub_name}", 1)
        elif isinstance(new_track_string, list):
            for x in new_track_string:
                if x not in self.track_strings:
                    self.add_track_string(x)
        else:
            print("[ERROR] track_string must be a string or a list of strings")
    def remove_track_string(self, track_string):
        if isinstance(track_string, str) and track_string in self.track_strings:
            self.track_strings.remove(track_string)
            marker_pub = self.marker_publishers[track_string]
            pc_pub = self.pc_publishers[track_string]
            self.destroy_publisher(marker_pub)
            self.destroy_publisher(pc_pub)
            del self.marker_publishers[track_string]
            del self.marker_publishers[track_string]

        elif isinstance(track_string, list):
            for x in track_string:
                self.remove_track_string(x)
        else:
            print("[ERROR] track_string must be a string or a list of strings")
 

    def _publish_loop(self, rate_hz):
        while rclpy.ok():
            if self.new_data:
                for query in self.track_strings:
                    pc_msg = self.get_pointcloud_msg(query)
                    marker_msg = self.get_marker_msg(query)
                    self.pc_publishers[query].publish(pc_msg)
                    self.marker_publishers[query].publish(marker_msg)
                with self._lock:
                    self.new_data = False
            time.sleep(1.0 / rate_hz)

    def get_pointcloud(self, query):
        #print(f"get_current_pointcloud called")
        pcd = o3d.geometry.PointCloud()
        if query not in self.tracked_objects:
            print(f"[ERROR] No tracked objects found for query: {query}")

        pcds = self.tracked_objects[query]["pcds"]
        for p in pcds:
            pcd += p

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        rgb = (colors * 255).astype(np.uint8)
        nx6 = np.hstack([points, rgb])
        return nx6
    
    def get_pointcloud_msg(self, query):
        points = self.get_pointcloud(query)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "pelvis"  # change as needed

        fields = []
        for name, offset in zip(['x', 'y', 'z', 'rgb'], [0, 4, 8, 12]):
            f = PointField()
            f.name = name
            f.offset = offset
            f.datatype = PointField.FLOAT32
            f.count = 1
            fields.append(f)
        # Convert rgb from uint8 to packed float
        rgb_float = (
            (points[:, 3].astype(np.uint32) << 16) |
            (points[:, 4].astype(np.uint32) << 8) |
            (points[:, 5].astype(np.uint32))
        ).view(np.float32)
        pc = np.column_stack((points[:, :3], rgb_float))

        msg = point_cloud2.create_cloud(header, fields, pc)
        return msg
    
    def get_marker_msg(self, query):
        marker_array = MarkerArray()
        id = 0
        for box, score in zip(self.tracked_objects[query]["boxes"], self.tracked_objects[query]["scores"]):
            #print(f"Processing box for query: {query}, score: {score}")
            #input(f"{box=}, {score=}")
            score = score.item()
            r = 1-score
            g = score
            #print(type(r), type(g))
            #input(f"{r=}, {g=}")
            
            box_marker = Marker()
            box_marker.header.frame_id = "pelvis"   # or your preferred frame
            box_marker.ns = "boxes"
            box_marker.id = 0
            box_marker.type = Marker.LINE_LIST
            box_marker.action = Marker.ADD
            box_marker.pose.orientation.w = 1.0  # No rotation
            box_marker.scale.x = 0.01            # Line width in meters
            box_marker.color.r = r*0.5
            box_marker.color.g = g*0.5
            box_marker.color.b = 0.0
            box_marker.color.a = 0.5
            box_marker.id = id
            id += 1
            #print(f"{dir(box)=}")
            #input("Press Enter to continue...")
            box_marker.points = box_to_points(box)
            marker_array.markers.append(box_marker)


            score_marker = Marker()
            score_marker.header.frame_id = "head_camera_link"
            score_marker.ns = "scores"
            score_marker.id = id
            id += 1
            score_marker.type = Marker.TEXT_VIEW_FACING
            score_marker.action = Marker.ADD
            #print(f"{dir(box)=}")
            center = box.get_center()
            #print(f"{center=}")
            #input("Press Enter to continue...")

            score_marker.pose.position.x = center[0]
            score_marker.pose.position.y = center[1]
            score_marker.pose.position.z = center[2] # Slightly above the box
            score_marker.pose.orientation.w = 1.0
            score_marker.scale.z = 0.05  # Font size
            score_marker.color.r = r
            score_marker.color.g = g
            score_marker.color.b = 0.0
            score_marker.color.a = 1.0
            score_marker.text = f"{query.replace(' ', '')}:{score:.2f}"
            #print(f"{score_marker.text=}")
            marker_array.markers.append(score_marker)
        return marker_array
        

    def update(self, debug=False):
        rgb_img, depth_img, info = self.sub.get_data()
        pose = [0,0,0,0,0,0]
        if rgb_img is None or depth_img is None or info is None:
            print("No image received yet.")
            return False
        if len(self.track_strings) == 0:
            #print("No track strings provided.")
            return False
        #print(f"\n\n{info=}")
        intrinsics = {
            "fx": info.k[0],
            "fy": info.k[4],
            "cx": info.k[2],
            "cy": info.k[5],
            "width": info.width,
            "height": info.height
        }
        result = super().update(rgb_img, depth_img, self.track_strings, intrinsics, pose, debug=debug)
        with self._lock:
            self.new_data = result
        return result

    
            
def TestVisionPipe(args=None):
    rclpy.init(args=args)
    VP = ROS_VisionPipe()
    VP.add_track_string("drill")
    VP.add_track_string("wrench")
    VP.add_track_string("soda can")
    VP.add_track_string(["wrench", "screwdriver"])
    success_counter = 0
    while rclpy.ok():
        #print("looped")
        success = VP.update()
        success_counter += 1 if success != False else 0
        if success:
            print(f"Success {success_counter} with {len(VP.tracked_objects)} tracked objects")
            #VP.vis_belief2D(query=f"{VP.track_strings[-1]}", blocking=False, prefix = f"T={success_counter} ")
            pass
        time.sleep(1)


    for object, predictions in VP.tracked_objects.items():
        print(f"{object=}")
        print(f"   {len(predictions['boxes'])=}, {len(predictions['pcds'])=}, {predictions['scores'].shape=}")
        for i, pcd in enumerate(predictions["pcds"]):
            print(f"   {i=}, {predictions['scores'][i]=}")
        print(f"{object=}")
        VP.vis_belief2D(query=object, blocking=False, prefix=f"Final {object} predictions")