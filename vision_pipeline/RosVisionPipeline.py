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
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time

from rclpy.duration import Duration

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2


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
            score_marker.header.frame_id = "pelvis"
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
        rgb_img, depth_img, info, pose = self.sub.get_data()
        if pose is None:
            print("No pose received yet.")
            return False
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