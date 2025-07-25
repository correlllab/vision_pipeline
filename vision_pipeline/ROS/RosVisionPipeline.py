#!/usr/bin/env python3
import sys
import threading
import time
import open3d as o3d
import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import sys
import json

from sensor_msgs.msg  import PointCloud2, PointField
from sensor_msgs_py   import point_cloud2
from std_msgs.msg import Header
from rclpy.time import Time
from visualization_msgs.msg import Marker, MarkerArray
from custom_ros_messages.srv import Query, UpdateTrackedObject


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
from VisionPipeline import VisionPipe
from RosRealsense import RealSenseSubscriber
from ros_utils import box_to_points, pcd_to_msg



class ROS_VisionPipe(VisionPipe, Node):
    def __init__(self, subscribers):
        assert isinstance(subscribers, (RealSenseSubscriber, list)), "Subscribers must be a RealSenseSubscriber or a list of them."
        if isinstance(subscribers, RealSenseSubscriber):
            subscribers = [subscribers]
        VisionPipe.__init__(self)
        Node.__init__(self, "ros_vision_pipe")
        self.subscribers = subscribers
        self.track_strings = []
        self.marker_publishers = {}
        self.pc_publishers = {}
        self._lock = threading.RLock()
        self.new_data = False
        self.vis_frame = "pelvis"
        self.start_services()
    def start_services(self) -> None:
        """
        Advertise the two custom services and start a background spin thread.
        Using a single MultiThreadedExecutor avoids the “node added to more
        than one executor” error and still lets multiple callbacks run in
        parallel.
        """
        # ── Advertise services ───────────────────────────────────────────────
        self.update_srv = self.create_service(
            UpdateTrackedObject,
            'vp_update_tracked_object',
            self.update_track_string_callback,
        )
        self.query_srv = self.create_service(
            Query,
            'vp_query_tracked_objects',
            self.query_tracked_objects_callback,
        )

        # ── Executor in its own thread ───────────────────────────────────────
        self._executor = rclpy.executors.MultiThreadedExecutor()
        self._executor.add_node(self)

        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True,
            name='ros_vision_pipe_spin',
        )
        self._spin_thread.start()

    def update_track_string_callback(self, request, response):
        print(f"Received request to {request.action} track string: {request.object}")
        if request.action == "add":
            success = self.add_track_string(request.object)
            response.result = success
            if success:
                response.message = f"Added track string: {request.object}"
            else:
                response.message = f"Failed to add track string: {request.object}"
        elif request.action == "remove":
            success = self.remove_track_string(request.object)
            response.result = success
            if success:
                response.message = f"Removed track string: {request.object}"
            else:
                response.message = f"Failed to remove track string: {request.object}"
        else:
            response.result = False
            response.message = "Invalid action. Use 'add' or 'remove'."
        return response

    def query_tracked_objects_callback(self, request, response):
        #print(f"Querying tracked objects for: {request.query}")
        with self._lock:
            print("Acquired lock for querying tracked objects.")
            print(f"{request.query=} {request.query not in self.tracked_objects=}")
            if request.query not in self.tracked_objects:
                response.cloud = PointCloud2()
                response.result = False
                response.message = f"No tracked objects found for query: {request.query}"
                return response

            top_pcd, prob = self.query(request.query)
            response.cloud = pcd_to_msg(top_pcd, self.vis_frame)
            response.prob = float(prob.item())
            response.result = True
            response.message = f"Tracked objects found for query: {request.query} with prob {prob:.2f}"
        return response

    def add_track_string(self, new_track_string):
        with self._lock:
            #print(f"Adding track string: {new_track_string} lock aquired")
            if isinstance(new_track_string, str) and new_track_string not in self.track_strings:
                self.track_strings.append(new_track_string)
                topic_sub_name = new_track_string.replace(" ", "_")
                self.marker_publishers[new_track_string] = self.create_publisher(MarkerArray, f"/tracked_objects/markers/{topic_sub_name}", 1)
                self.pc_publishers[new_track_string] = self.create_publisher(PointCloud2, f"/tracked_objects/pointcloud/{topic_sub_name}", 1)
                return True
            elif isinstance(new_track_string, list):
                for x in new_track_string:
                    if x not in self.track_strings:
                        self.add_track_string(x)
                return True
            else:
                print("[ERROR] track_string must be a string or a list of strings")
                return False
    def remove_track_string(self, track_string):
        with self._lock:
            #print(f"Removing track string: {track_string} lock aquired")
            if isinstance(track_string, str) and track_string in self.track_strings:
                self.track_strings.remove(track_string)
                marker_pub = self.marker_publishers[track_string]
                pc_pub = self.pc_publishers[track_string]
                self.destroy_publisher(marker_pub)
                self.destroy_publisher(pc_pub)
                del self.marker_publishers[track_string]
                del self.pc_publishers[track_string]
                return True
            elif isinstance(track_string, list):
                for x in track_string:
                    self.remove_track_string(x)
                return True
            else:
                print("[ERROR] track_string must be a string or a list of strings")
                return False
    def publish_viz(self):
        with self._lock:
            if self.new_data:
                for query in self.tracked_objects:
                    candidate_set = None
                    top_candidate_set = {"probs":[], "boxes":[], "pcds":[] }

                    candidates = self.tracked_objects[query]
                    values, indicies = candidates["probs"].topk(min(config["vis_k"], len(candidates["probs"])), largest=True)
                    for i in indicies.tolist():
                        top_candidate_set["probs"].append(candidates["probs"][i])
                        top_candidate_set["boxes"].append(candidates["boxes"][i])
                        top_candidate_set["pcds"].append(candidates["pcds"][i])
                    pcd = self.get_pointcloud(top_candidate_set)
                    pc_msg = pcd_to_msg(pcd, self.vis_frame)
                    self.pc_publishers[query].publish(pc_msg)

                    marker_msg = self.get_marker_msg(top_candidate_set, query)
                    self.marker_publishers[query].publish(marker_msg)
                self.new_data = False
    def get_pointcloud(self, candidates_set):
        #print(f"get_current_pointcloud called")
        pcd = o3d.geometry.PointCloud()
        pcds = None
        for p in candidates_set["pcds"]:
            pcd += p
        return pcd



    def get_marker_msg(self, candidate_set, query):
        marker_array = MarkerArray()
        id = 0
        for box, prob in zip(candidate_set["boxes"], candidate_set["probs"]):
            #print(f"Processing box for query: {query}, score: {score}")
            #input(f"{box=}, {score=}")
            prob = prob.item()
            r = 1-prob
            g = prob
            
            #print(type(r), type(g))
            #input(f"{r=}, {g=}")

            box_marker = Marker()
            box_marker.header.frame_id = self.vis_frame   # or your preferred frame
            box_marker.ns = "boxes"
            box_marker.id = id
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


            prob_marker = Marker()
            prob_marker.header.frame_id = self.vis_frame
            prob_marker.ns = "probs"
            prob_marker.id = id
            id += 1
            prob_marker.type = Marker.TEXT_VIEW_FACING
            prob_marker.action = Marker.ADD
            #print(f"{dir(box)=}")
            center = box.get_center()
            #print(f"{center=}")
            #input("Press Enter to continue...")

            prob_marker.pose.position.x = center[0]
            prob_marker.pose.position.y = center[1]
            prob_marker.pose.position.z = center[2] # Slightly above the box
            prob_marker.pose.orientation.w = 1.0
            prob_marker.scale.z = 0.05  # Font size
            prob_marker.color.r = r
            prob_marker.color.g = g
            prob_marker.color.b = 0.0
            prob_marker.color.a = 1.0
            prob_marker.text = f"{query.replace(' ', '')}:{prob:.2f}"
            #print(f"{score_marker.text=}")
            marker_array.markers.append(prob_marker)
        return marker_array

    def update(self, debug=False):
        update_success = []
        for sub in self.subscribers:
            rgb_img, depth_img, info, pose = sub.get_data()
            success = True
            msg_components = []
            if pose is None:
                msg_components.append("no pose data")
                success = success and False
            if rgb_img is None or depth_img is None or info is None:
                msg_components.append("no image data")
                success = success and False
            if len(self.track_strings) == 0:
                msg_components.append("no track strings")
                success = success and False

            if not success:
                msg = ", ".join(msg_components)
                update_success.append((sub.camera_name, False, msg))
                continue

            intrinsics = {
                "fx": info.k[0],
                "fy": info.k[4],
                "cx": info.k[2],
                "cy": info.k[5],
                "width": info.width,
                "height": info.height
            }
            with self._lock:
                time_stamp = Time.from_msg(info.header.stamp).nanoseconds / 1e9
                result, update_msg = super().update(rgb_img, depth_img, self.track_strings, intrinsics, pose, time_stamp, debug=debug)
                self.new_data = self.new_data or result
                msg_components.append(update_msg)
            msg = ", ".join(msg_components)
            update_success.append((sub.camera_name, result, msg))
        return update_success

def RunVisionPipe():
    rclpy.init()
    subs = []
    for name in config["rs_names"]:
        sub = RealSenseSubscriber(name)
        subs.append(sub)
    VP = ROS_VisionPipe(subs)
    try:
        while rclpy.ok():
            success = VP.update(debug=config["debug"])
            out_str = ""
            for cam_name, result, msg in success:
                if result:
                    out_str += f"   {cam_name}: OK ({msg})\n"
                else:
                    out_str += f"   {cam_name}: FAILED ({msg})\n"
            print(f"{len(VP.track_strings)=} Update Status:\n{out_str}")
            VP.publish_viz()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        VP.destroy_node()
        rclpy.shutdown()

class ExampleClient(Node):
    def __init__(self):
        super().__init__('dummy_client')
        self.update_client = self.create_client(UpdateTrackedObject, 'vp_update_tracked_object')
        self.query_client = self.create_client(Query, 'vp_query_tracked_objects')
        while not self.update_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Update Service not available, waiting again...')
        while not self.query_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Query Service not available, waiting again...')
        self.pc_pub = self.create_publisher(PointCloud2, '/tracked_objects/pointcloud', 1)
        print("ExampleClient initialized and services are available.")
    def add_track_string(self, track_string):
        req = UpdateTrackedObject.Request()
        req.object = track_string
        req.action = "add"
        future = self.update_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def remove_track_string(self, track_string):
        req = UpdateTrackedObject.Request()
        req.object = track_string
        req.action = "remove"
        future = self.update_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def query_tracked_objects(self, track_string):
        req = Query.Request()
        req.query = track_string
        future = self.query_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        self.pc_pub.publish(result.cloud)
        return result

def TestExampleClient(args=None):
    rclpy.init()
    ec = ExampleClient()
    try:
        while rclpy.ok():
            action = int(input("Enter an integer for the action (1: add, 2: remove, 3: query): "))
            if action not in [1, 2, 3]:
                print("Invalid action. Please enter 1, 2, or 3")
                continue
            if action == 1:
                track_string = input("Enter the track string to add: ")
                ats_out = ec.add_track_string(track_string)
                print(f"add_track_string response: {ats_out}\n")
            elif action == 2:
                track_string = input("Enter the track string to remove: ")
                dts_out = ec.remove_track_string(track_string)
                print(f"remove_track_string response: {dts_out}\n")
            elif action == 3:
                track_string = input("Enter the track string to query: ")
                print(f"Querying tracked objects for '{track_string}'...")
                q_out = ec.query_tracked_objects(track_string)
                print(f"query_tracked_objects response: {q_out.message}\n")
            time.sleep(1)

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        ec.destroy_node()
        rclpy.shutdown()
        return 0
