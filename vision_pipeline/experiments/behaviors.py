import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from custom_ros_messages.srv import Query, UpdateTrackedObject, UpdateBeliefs, ResetBeliefs
from custom_ros_messages.action import DualArm
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R
import time
from unitree_go.msg import MotorCmds, MotorCmd
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np

def pose_to_matrix(pose_array):
    x, y, z, roll, pitch, yaw = pose_array
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    matrix = np.eye(4)
    matrix[:3, :3] = r.as_matrix()
    matrix[:3, 3] = [x, y, z]
    return matrix

class BehaviorNode(Node):
    def __init__(self, vp = True):
        rclpy.init()
        super().__init__('coordinator')
        if vp:
            self.update_tracked_client = self.create_client(UpdateTrackedObject, 'vp_update_tracked_object')
            self.query_client = self.create_client(Query, 'vp_query_tracked_objects')
            self.update_belief_client = self.create_client(UpdateBeliefs, "vp_update_beliefs")
            self.reset_beliefs_client = self.create_client(ResetBeliefs, "vp_reset_beliefs")
            self.forget_everything_client = self.create_client(ResetBeliefs, "vp_forget_everything")
            while not self.update_tracked_client.wait_for_service(timeout_sec=1.0):
                print('Update Tracked Service not available, waiting again...')
            while not self.query_client.wait_for_service(timeout_sec=1.0):
                print('Query Service not available, waiting again...')
            while not self.update_belief_client.wait_for_service(timeout_sec=1.0):
                print("Belief update service not available")
            while not self.reset_beliefs_client.wait_for_service(timeout_sec=1.0):
                print("Reset beliefs service not available")
            while not self.forget_everything_client.wait_for_service(timeout_sec=1.0):
                print("Forget everything service not available")

        self.action_client = ActionClient(
            self,
            DualArm,
            'move_dual_arm'
        )
        lM = np.eye(4)
        lM[:3, 3] = [0.3, 0.5, 0.2]
        rM = np.eye(4)
        rM[:3, 3] = [0.3, -0.5, 0.2]
        self.r_arm_mat = rM
        self.l_arm_mat = lM
        self.marker_pub = self.create_publisher(Marker, "/camera_marker", 10)


        self.hand_pub = self.create_publisher(MotorCmds, '/inspire/cmd', 10)
        self.hand_length = 0.3
        
        self.l_hand = None
        self.r_hand = None

        PC_QOS = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pointcloud_pub = self.create_publisher(PointCloud2, "/experiment_pointcloud", PC_QOS)


        
        self.go_home()
        self.open_hands()
        time.sleep(1)
        self.close_hands()
        time.sleep(1)

    def publish_pointcloud(self, cloud_msg):
        self.pointcloud_pub.publish(cloud_msg)
        print(f"point cloud published")

    def publish_marker(self, x,y,z):
        marker = Marker()

        marker.header.frame_id = "pelvis"
        marker.ns = "behavior marker"

        marker.type = Marker.DELETEALL
        self.marker_pub.publish(marker)



        marker.type = Marker.SPHERE
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)

    def reset_beliefs(self):
        request = ResetBeliefs.Request()
        print(f"sending reset request")
        future = self.reset_beliefs_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        print(f"{result.message=}, {result.success=}")
    
    def forget_everything(self):
        request = ResetBeliefs.Request()
        print(f"sending forget request")
        future = self.forget_everything_client.call_async(request)
        print("request sent")
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        print(f"{result.message=}, {result.success=}")

    def update_head(self):
        self.update_belief("/realsense/head")
    def update_hand(self):
        self.update_belief("/realsense/left_hand")

    def update_belief(self, namespace):
        req = UpdateBeliefs.Request()
        req.camera_name_space = namespace
        print("sending belief update request")
        future = self.update_belief_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        print(f"[update belief] {result.success=}")
    def set_hands(self, l_goal=None, r_goal=None):
        if l_goal is not None:
            self.l_hand = l_goal
        if r_goal is not None:
            self.r_hand = r_goal

        msg = MotorCmds()
        msg.cmds = [MotorCmd(mode=1, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0) for i in range(12)]
        for i, (l_q, r_q) in enumerate(zip(self.l_hand, self.r_hand)):
            msg.cmds[i].q = r_q
            msg.cmds[6+i].q = l_q

        self.hand_pub.publish(msg)

    def open_hands(self):
        l_hand = [1.0]*6
        r_hand = [1.0]*6
        self.set_hands(l_goal = l_hand, r_goal = r_hand)
    def close_hands(self):
        l_hand = [0.0]*6
        l_hand[5] = 1.0
        r_hand = [0.0]*6
        r_hand[5] = 1.0
        self.set_hands(l_goal = l_hand, r_goal = r_hand)

    def go_home(self):
        goal_msg = DualArm.Goal()
        goal_msg.duration = 10
        goal_msg.keyword = "home"

        self.action_client.wait_for_server()

        # send action
        print('Going home...')
        self.start_time = time.time()
        future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected')
            return

        future_result = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, future_result)
        result = future_result.result().result
        print()
        print(f'Final result: success = {result.success}')
        time.sleep(1)
   
    def track_object(self, obj_name):
        req = UpdateTrackedObject.Request()
        req.object = obj_name
        req.action = "add"
        print("sending tracked object request")
        future = self.update_tracked_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        return result

    def query_objects(self, query, threshold, specific_name=""):
        req = Query.Request()
        req.query = query
        req.confidence_threshold = threshold
        req.pc_name = specific_name
        print("sending query request")
        future = self.query_client.call_async(req)
        print("query sent")
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        print(f"{query=} result: {result.success} {len(result.probabilities)=} {len(result.clouds)=} {len(result.names)=} {result.message=}")
        return result

    def send_arm_goal(self, left_mat = None, right_mat = None, duration=3, block = True):
        assert left_mat is None or left_mat.shape == (4,4)
        assert right_mat is None or right_mat.shape == (4,4)
        assert duration > 0 and isinstance(duration, int)
        if left_mat is not None:
            self.l_arm_mat = left_mat
        if right_mat is not None:
            self.r_arm_mat = right_mat

        goal_msg = DualArm.Goal()
        goal_msg.left_target = self.l_arm_mat.reshape(-1).tolist()
        goal_msg.right_target = self.r_arm_mat.reshape(-1).tolist()
        goal_msg.duration = duration

        self.action_client.wait_for_server()

        # send action
        print('Sending goal...')
        self.start_time = time.time()
        future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected')
            return

        # start a cancel listener thread
        # print('Goal accepted, waiting for result...')

        if block:
            # wait till finish
            future_result = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, future_result)
            result = future_result.result().result
            time.sleep(1)
        print()
        print(f'Final result: success = {result.success}')
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        print(f'\rLeft Error Linear: {feedback.left_error_linear:.2f} Angular: {feedback.left_error_angular:.2f}; Right Error Linear: {feedback.right_error_linear:.2f} Angular: {feedback.right_error_linear:.2f} T:{time.time()- self.start_time:.2f}', end="", flush=True)

    def close(self):
        self.destroy_node()
        rclpy.shutdown()