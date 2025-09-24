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
from custom_ros_messages.srv import Query, UpdateTrackedObject, UpdateBeliefs, ResetBeliefs
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


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
from config import config
if ros_dir not in sys.path:
    sys.path.insert(0, ros_dir)
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)
from VisionPipeline import VisionPipe
from RosRealsense import RealSenseSubscriber
from ros_utils import box_to_marker, text_marker, pcd_to_msg

from rclpy.executors import MultiThreadedExecutor

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
        self.vis_frame = config["base_frame"]
        # self.create_timer(1.0/6.0, self.publish_viz)

        self.next_marker_id = 1
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.start_services()
        self.add_track_string(config["test_querys"])
    def start_services(self) -> None:
        
        self.update_tracked_srv = self.create_service(
            UpdateTrackedObject, 'vp_update_tracked_object',
            self.update_track_string_callback
        )
        self.query_srv = self.create_service(
            Query, 'vp_query_tracked_objects',
            self.query_tracked_objects_callback
        )

        self.update_belief_srv = self.create_service(
            UpdateBeliefs, 'vp_update_beliefs',
            self.update_beliefs_callback
        )

        self.reset_belief_srv = self.create_service(
            ResetBeliefs, 'vp_reset_beliefs',
            self.reset_probs_callback
        )

        self.forget_everything_srv = self.create_service(
            ResetBeliefs, 'vp_forget_everything',
            self.forget_everything_callback
        )

    def update_track_string_callback(self, request, response):
        print(f"\n\nReceived request to {request.action} track string: {request.object}")
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

    def reset_probs_callback(self, request, response):
        print("\n\nResetting probs...")
        with self._lock:
            print("[ROS VP reset probs] lock aquired")
            for tracked_obj in self.tracked_objects:
                print(f"Clearing tracked object: {tracked_obj}")
                self.tracked_objects[tracked_obj]["probs"] = [0.5 for _ in self.tracked_objects[tracked_obj]["probs"]]
            print("[ROS VP reset probs] lock released")
            
        print("Vision pipeline beliefs has been reset.")
        response.success = True
        response.message = "Vision pipeline beliefs reset successfully."
        return response
    
    def forget_everything_callback(self, request, response):
        print("\n\nForgetting everything...")
        with self._lock:
            print("[ROS VP forget everything] lock aquired")
            self.tracked_objects = {}
            self.pose_time_tracker = []
            self.update_count = 1

            self.track_strings = []
            self.next_marker_id = 1
            for pub in self.marker_publishers.values():
                self.destroy_publisher(pub)
            for pub in self.pc_publishers.values():
                self.destroy_publisher(pub)
            self.marker_publishers = {}
            self.pc_publishers = {}
            print("[ROS VP forget everything] lock released")

        self.add_track_string(config["test_querys"])
        print("Vision pipeline has been reset.")
        response.success = True
        response.message = "Vision pipeline forget everything successfully."
        return response

    def query_tracked_objects_callback(self, request, response):
        print(f"\n\nQuerying tracked objects for: {request.query}")
        
        print(f"{request.query=} {request.query.lower() not in self.tracked_objects=}")
        q = request.query.lower()
        if q not in self.tracked_objects:
            response.clouds = [PointCloud2()]
            response.probabilities = [0.0]
            response.names = [""]
            response.success = False
            response.message = f"No tracked objects found for query: {q}"
            return response
        with self._lock:
            print("[ROS VP query tracked] lock aquired")
            pcd_arr, prob_arr, name_arr = self.query(q, request.confidence_threshold)
            print("[ROS VP query tracked] lock released")
        if request.pc_name != "":
            idx = name_arr.index(request.pc_name) if request.pc_name in name_arr else -1
            if idx == -1:
                response.clouds = [PointCloud2()]
                response.probabilities = [0.0]
                response.names = [""]
                response.success = False
                response.message = f"No tracked objects found for query: {q} with name: {request.pc_name}"
                return response
            pcd_arr = [pcd_arr[idx]]
            prob_arr = [prob_arr[idx]]
            name_arr = [name_arr[idx]]

        response.clouds = [pcd_to_msg(pcd, self.vis_frame) for pcd in pcd_arr]
        response.probabilities = prob_arr
        response.success = True
        response.names = name_arr
        response.message = f"Tracked objects found for query: {len(pcd_arr)= } with probs {len(prob_arr)=} and names {name_arr=}"
        print(response.message)
        return response

    def update_beliefs_callback(self, request, response):
        print(f"\n\nReceived request to update beliefs using camera: {request.camera_name_space}")
        sub = None
        for s in self.subscribers:
            if s.camera_name_space == request.camera_name_space:
                sub = s
        if sub is None:
            response.message = f"No subscriber found for camera: {request.camera_name_space}"
            response.success = False
            return response
        print(f"Updating beliefs using camera: {request.camera_name_space} waiting for lock...")
        with self._lock:
            print("[ROS VP update beliefs] lock aquired retrieve data")
            rgb_img, depth_img, info, pose = sub.get_data()
            print("[ROS VP update beliefs] lock released retrieve data")
        msg_components = [f"{request.camera_name_space}: "]
        success = True
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
            response.message = msg
            response.success = False
            return response
        
        intrinsics = {
            "fx": info.k[0],
            "fy": info.k[4],
            "cx": info.k[2],
            "cy": info.k[5],
            "width": info.width,
            "height": info.height
        }
        time_stamp = Time.from_msg(info.header.stamp).nanoseconds / 1e9
        with self._lock:
            print("[ROS VP update beliefs] lock aquired")
            result, update_msg = super().update(rgb_img, depth_img, self.track_strings, intrinsics, pose, time_stamp, debug=False)
            print("[ROS VP update beliefs] lock released")
        response.message = update_msg
        response.success = result

        return response

    def add_track_string(self, new_track_string):
        print(f"\n\nAdding track string: {new_track_string}")
       
        #print(f"Adding track string: {new_track_string} lock aquired")
        if isinstance(new_track_string, str) and new_track_string not in self.track_strings:
            new_track_string = new_track_string.lower()
            print("[ROS VP add track] waiting for lock")
            with self._lock:
                print("[ROS VP add track] lock aquired")
                self.track_strings.append(new_track_string)
                print("[ROS VP add track] lock released")
            topic_sub_name = new_track_string.replace(" ", "_")
            self.marker_publishers[new_track_string] = self.create_publisher(MarkerArray, f"/tracked_objects/markers/{topic_sub_name}", self.qos)
            self.pc_publishers[new_track_string] = self.create_publisher(PointCloud2, f"/tracked_objects/pointcloud/{topic_sub_name}", self.qos)
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
        print(f"\n\nRemoving track string: {track_string}")
        
        #print(f"Removing track string: {track_string} lock aquired")
        if isinstance(track_string, str) and track_string in self.track_strings:
            new_track_string = new_track_string.lower()

            print("[ROS VP remove track] waiting for lock")
            with self._lock:
                print("[ROS VP remove track] lock aquired")
                self.track_strings.remove(track_string)
                print("[ROS VP remove track] lock released")

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
            self.next_marker_id = 1
            for query in self.tracked_objects:
                candidate_set = None
                top_candidate_set = {"probs":[], "boxes":[], "pcds":[], "names":[]}

                candidates = self.tracked_objects[query]
                probs = np.array(candidates["probs"])
                # k = min(config["vis_k"], len(probs))
                # indices = np.argsort(probs)[-k:][::-1]  # descending order
                indices = range(len(probs))
                for i in indices:
                    top_candidate_set["probs"].append(candidates["probs"][i])
                    top_candidate_set["boxes"].append(candidates["boxes"][i])
                    top_candidate_set["pcds"].append(candidates["pcds"][i])
                    top_candidate_set["names"].append(candidates["names"][i])
                pcd = self.get_pointcloud(top_candidate_set)
                if pcd is None or len(pcd.point["positions"]) == 0:
                    continue
                # print(f"pcd_to_msg publish vis {type(pcd)=}")
                pc_msg = pcd_to_msg(pcd, self.vis_frame)
                self.pc_publishers[query].publish(pc_msg)


                clear_arr = MarkerArray()

                clear_marker = Marker()
                clear_marker.action = Marker.DELETEALL
                # (Optional, but avoids RViz warnings)
                clear_marker.header.stamp = self.get_clock().now().to_msg()
                clear_marker.header.frame_id = config["base_frame"]

                clear_arr.markers.append(clear_marker)
                self.marker_publishers[query].publish(clear_arr)

                marker_msg = self.get_marker_msg(top_candidate_set, query)
                self.marker_publishers[query].publish(marker_msg)
    def get_pointcloud(self, candidates_set):
        """
        Accumulate a list of o3d.t.geometry.PointCloud objects into one.
        Merges 'positions' and 'colors' if present.
        """
        pcd_acc = None

        for p in candidates_set["pcds"]:
            if pcd_acc is None:
                pcd_acc = p
            else:
                pcd_acc = pcd_acc.append(p)
        return pcd_acc
    def get_marker_msg(self, candidate_set, query):
        marker_array = MarkerArray()
        
        for i, (box, prob, name) in enumerate(zip(candidate_set["boxes"], candidate_set["probs"], candidate_set["names"])):
            #print(f"Processing box for query: {query}, score: {score}")
            #input(f"{box=}, {score=}")
            r = 1-prob
            g = prob
            
            #print(type(r), type(g))
            #input(f"{r=}, {g=}")
            box_marker = box_to_marker(box.to_legacy(), [r, g, 0.0, 0.5], self.vis_frame, self.next_marker_id)
            marker_array.markers.append(box_marker)
            self.next_marker_id += 1

            prob_marker = text_marker(f"{name}:{prob:.2f}",
                                       box.get_center().numpy().tolist(),
                                        [r, g, 0.0, 0.5], self.vis_frame, self.next_marker_id)
            marker_array.markers.append(prob_marker)
            self.next_marker_id += 1

        return marker_array


def RunVisionPipe():
    rclpy.init()

    # 1. Create an executor that can handle multiple nodes and their callbacks
    executor = MultiThreadedExecutor()
    
    subs = []
    # 2. Create your subscriber nodes and add them to the executor
    for name_space in config["rs_name_spaces"]:
        sub = RealSenseSubscriber(name_space)
        subs.append(sub)
        executor.add_node(sub)

    # 3. Create your main node and add it to the executor
    VP = ROS_VisionPipe(subs)
    VP.add_track_string(config["test_querys"])
    executor.add_node(VP)

    # 4. Spin the executor to run all nodes until shutdown
    try:
        executor.spin()
    finally:
        # Cleanly destroy nodes and shut down rclpy
        executor.shutdown()
        VP.destroy_node()
        for sub in subs:
            sub.destroy_node()
        rclpy.shutdown()


class ExampleClient(Node):
    def __init__(self):
        super().__init__('dummy_client')
        self.update_tracked_client = self.create_client(UpdateTrackedObject, 'vp_update_tracked_object')
        self.update_belief_client = self.create_client(UpdateBeliefs, 'vp_update_beliefs')
        self.query_client = self.create_client(Query, 'vp_query_tracked_objects')
        self.reset_belief_client = self.create_client(ResetBeliefs, 'vp_reset_beliefs')
        while not self.update_tracked_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Update  tracked Service not available, waiting again...')
        while not self.query_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Query Service not available, waiting again...')
        while not self.update_belief_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Update belief Service not available, waiting again...')
        while not self.reset_belief_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset belief Service not available, waiting again...')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.pc_pub = self.create_publisher(PointCloud2, '/tracked_objects/pointcloud', qos)
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
        return result
    
    def update_beliefs(self, camera_name_space):
        req = UpdateBeliefs.Request()
        req.camera_name_space = camera_name_space
        future = self.update_belief_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def reset_vp(self):
        req = ResetBeliefs.Request()
        future = self.reset_belief_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

def TestExampleClient(args=None):
    rclpy.init()
    ec = ExampleClient()
    try:
        while rclpy.ok():
            i = input("two words, the action, add, remove, belief, query, or reset (a, r, q, b, re): followed by string: ").lower().split()
            print(f"{i=}")
            action = i[0]
            track_string = " ".join(i[1:])
            print(f"{action=}, {track_string=}")
            if action not in ["add", "remove", "query", "belief", "reset", "a", "r", "q", "b", "re"]:
                print("Invalid action. Please enter add, remove, or query")
                continue
            if action == "a" or action == "add":
                ats_out = ec.add_track_string(track_string)
                print(f"add_track_string response: {ats_out}\n")
            elif action == "r" or action == "remove":
                dts_out = ec.remove_track_string(track_string)
                print(f"remove_track_string response: {dts_out}\n")
            elif action == "q" or action == "query":
                print(f"Querying tracked objects for '{track_string}'...")
                q_out = ec.query_tracked_objects(track_string)
                print(f"query_tracked_objects response: {q_out.message}\n")
            elif action == "b" or action == "belief":
                name_space = input("Enter the camera name space to update beliefs from: ")
                ub_out = ec.update_beliefs(name_space)
                print(f"update_beliefs response: {ub_out.message}, {ub_out.success}\n")
            elif action == "re" or action == "reset":
                print("Resetting vision pipeline...")
                re_out = ec.reset_vp()
                print(f"Vision pipeline reset. {re_out.message} {re_out.success}\n")
            time.sleep(1)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Shutting down...")
        ec.destroy_node()
        rclpy.shutdown()
        return 0
