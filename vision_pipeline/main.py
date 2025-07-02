import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from custom_ros_messages.srv import Query, UpdateTrackedObject
from custom_ros_messages.action import DualArm
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
import time
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
        self.update_client = self.create_client(UpdateTrackedObject, 'vp_update_tracked_object')
        self.query_client = self.create_client(Query, 'vp_query_tracked_objects')
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

        print("Sending goal to dual arm action server...")
        print(f"Left target: {left_target}")
        print(f"Right target: {right_target}")

def main():
    rclpy.init()
    node = main_node()
    objects = ["drill", "screwdriver", "wrench", "scissors", "soda can"]
    for obj in objects:
        result = node.track_object(obj)
        print(f"Tracking {obj}: {result}")
    time.sleep(3)

