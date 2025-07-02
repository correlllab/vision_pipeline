import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from custom_ros_messages.srv import Query, UpdateTrackedObject
from custom_ros_messages.action import DualArm
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
import time
from pynput import keyboard

import os
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from utils import msg_to_pcd


def pose_array_to_message(pose_array):
        pose = Pose()
        pose.position.x = pose_array[0]
        pose.position.y = pose_array[1]
        pose.position.z = pose_array[2]
        quat = R.from_euler('xyz', pose_array[3:], degrees=True).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        return pose

class main_node(Node):
    def __init__(self):
        super().__init__('coordinator')
        self.update_client = self.create_client(UpdateTrackedObject, 'vp_update_tracked_object')
        self.query_client = self.create_client(Query, 'vp_query_tracked_objects')
        while not self.update_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Update Service not available, waiting again...')
        while not self.query_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Query Service not available, waiting again...')
        self.action_client = ActionClient(
            self,
            DualArm,
            'move_dual_arm'
        )

    def track_object(self, obj_name):
        req = UpdateTrackedObject.Request()
        req.object = obj_name
        req.action = "add"
        future = self.update_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        return result

    def query_objects(self, query):
        req = Query.Request()
        req.query = query
        future = self.query_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        return result

    def send_goal(self, left_arr, right_arr):
        left_target = pose_array_to_message(left_arr)
        right_target = pose_array_to_message(right_arr)

        goal_msg = DualArm.Goal()
        goal_msg.left_target = left_target
        goal_msg.right_target = right_target

        self.action_client.wait_for_server()

        # send action
        self.get_logger().info('Sending goal...')
        future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        rclpy.spin_until_future_complete(self, future)
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().warn('Goal was rejected')
            return

        # start a cancel listener thread
        self.get_logger().info('Goal accepted, waiting for result...')
        self.get_logger().info('Press BACKSPACE to cancel the goal')
        listener = keyboard.Listener(
            on_press=self._keyboard_cancel
        )
        listener.start()

        # wait till finish
        future_result = self.goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, future_result)
        result = future_result.result().result
        self.get_logger().info(f'Final result: success = {result.success}')
        # stop the cancel listener thread
        listener.stop()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Left Error: {feedback.left_error:.2f}; Right Error: {feedback.right_error:.2f}')

    def _keyboard_cancel(self, key):
        if key == keyboard.Key.backspace:
            if self.goal_handle is not None:
                self.get_logger().info('Cancelling goal...')
                self.goal_handle.cancel_goal_async()

def main():
    rclpy.init()
    node = main_node()
    objects = ["drill", "screwdriver", "wrench", "scissors", "soda can"]
    for obj in objects:
       result = node.track_object(obj)
       print(f"Tracking {obj}: {result}")
    time.sleep(3)

    max_tries = 5
    tries = 0
    success = False
    query = None
    while success == False and tries < max_tries:
        query = node.query_objects("drill")
        success = query.result
        print(f"Query status: {query.message}")
        tries += 1
        time.sleep(3)
    if not success:
        print("Failed to query tracked objects after maximum tries.")
        return
    print(f"Query result: {query}")

    pcd = msg_to_pcd(query.cloud)
    center = pcd.get_center()
    print(f"Point cloud center: {center}")

    r_hand_goal = [0.3, -0.2, 0.1, 0, 0, 0]
    l_hand_goal = [center[0], center[1], center[2]+0.1, 0, 0, 0]
    res = node.send_goal(
        [0.5, 0.2, 0.2, 0.0, 0.0, 0.0],
        [0.5, -0.2, 0.2, 0.0, 0.0, 0.0]
    )
    print(f"Action result: {res}")

    node.destroy_node()
    rclpy.shutdown()
