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


class BehaviorNode(Node):
    def __init__(self):
        rclpy.init()
        super().__init__('coordinator')
        self.update_tracked_client = self.create_client(UpdateTrackedObject, '/vp_update_tracked_object')
        self.query_client = self.create_client(Query, '/vp_query_tracked_objects')
        self.update_belief_client = self.create_client(UpdateBeliefs, "/vp_update_beliefs")
        self.reset_beliefs_client = self.create_client(ResetBeliefs, "/vp_reset_beliefs")
        self.forget_everything_client = self.create_client(ResetBeliefs, "/vp_forget_everything")
        while not self.update_tracked_client.wait_for_service(timeout_sec=1.0):
            print('Update Tracked Service not available, waiting again...')
        while not self.query_client.wait_for_service(timeout_sec=1.0):
            print('Query Service not available, waiting again...')
        while not self.update_belief_client.wait_for_service(timeout_sec=1.0):
            print("Belief update service not available")
        while not self.reset_beliefs_client.wait_for_service(timeout_sec=1.0):
            print("Reset belief update service not available")
        while not self.forget_everything_client.wait_for_service(timeout_sec=1.0):
            print("Forget Everything service not available")

        self.action_client = ActionClient(
            self,
            DualArm,
            'move_dual_arm'
        )
        self.hand_pub = self.create_publisher(MotorCmds, '/inspire/cmd', 10)
        self.marker_pub = self.create_publisher(Marker, "/camera_marker", 10)


        PC_QOS = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pointcloud_pub = self.create_publisher(PointCloud2, "/experiment_pointcloud", PC_QOS)


        self.hand_length = 0.3
        self.r_arm_goal = None
        self.l_arm_goal = None
        self.l_hand = None
        self.r_hand = None
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
        
        res = self.send_arm_goal(
            left_arr = [0.3, 0.5, 0.2, 0, 0, 0],
            right_arr = [0.3, -0.5, 0.2, 0, 0, 0]
        )
        self.close_hands()
        time.sleep(1)

    def point_camera(self, x, y, z, height):
        self.publish_marker(x,y,z)
        safe_height = 0.2
        view_height = height + self.hand_length
        x_offset = -0.1
        new_x = x + x_offset
        new_y = y
        new_z = z + view_height
        roll = 90
        pitch = 90
        yaw = 0

        ready_goal = self.l_arm_goal
        ready_goal[0] = new_x
        ready_goal[1] = new_y
        ready_goal[2] = new_z
        ready_goal[3] = roll
        ready_goal[4] = pitch
        ready_goal[5] = yaw

        if self.l_arm_goal[3] != 90 and self.l_arm_goal[4] != 90:
            self.send_arm_goal(left_arr=ready_goal)
        
        final_goal = ready_goal
        final_goal[2] -= safe_height
        self.send_arm_goal(left_arr=final_goal)



    def point_finger(self, x,y,z):
        use_left_hand = y > 0

        #first raise and turn
        roll = 0
        pitch = 90
        yaw = 0
        if use_left_hand:
            roll = 90
            ready_goal = self.l_arm_goal
            ready_goal[2]+=0.3
            ready_goal[3]=roll
            ready_goal[4]=pitch
            ready_goal[5]=yaw
            self.send_arm_goal(left_arr = ready_goal)
        else:
            roll = 270

            ready_goal = self.r_arm_goal
            ready_goal[2]+=0.3
            ready_goal[3]=roll
            ready_goal[4]=pitch
            ready_goal[5]=yaw
            self.send_arm_goal(right_arr = ready_goal)


        #set hands
        if use_left_hand:
            self.set_hands(l_goal = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        else:
            self.set_hands(r_goal = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0])

        #finally go to pose
        x_new = x + 0.05
        y_new = y

        if use_left_hand:
            y_new += 0.05
        else:
            y_new -= 0.05
        
        z_new = z + hand_length

        if use_left_hand:
            final_goal = ready_goal
            final_goal[0] = x_new
            final_goal[1] = y_new
            final_goal[2] = z_new
            self.send_arm_goal(left_arr=final_goal)
        else:
            final_goal = ready_goal
            final_goal[0] = x_new
            final_goal[1] = y_new
            final_goal[2] = z_new
            self.send_arm_goal(right_arr=final_goal)
        
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
        print(f"{query=} result: {result.success} {len(result.probabilities)=} {len(result.clouds)=} {len(result.names)=}")
        return result

    def send_arm_goal(self, left_arr = None, right_arr = None):
        if left_arr is not None:
            self.l_arm_goal = left_arr
        if right_arr is not None:
            self.r_arm_goal = right_arr
        left_target = pose_array_to_message(self.l_arm_goal)
        right_target = pose_array_to_message(self.r_arm_goal)

        goal_msg = DualArm.Goal()
        goal_msg.left_target = left_target
        goal_msg.right_target = right_target

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


        # wait till finish
        future_result = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, future_result)
        result = future_result.result().result
        print(f'Final result: success = {result.success}')
        time.sleep(1)
        # print()
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # print(f'\rLeft Error Linear: {feedback.left_error_linear:.2f} Angular: {feedback.left_error_angular:.2f}; Right Error Linear: {feedback.right_error_linear:.2f} Angular: {feedback.right_error_linear:.2f} {time.time()- self.start_time:.2f}', end="", flush=True)

    def close(self):
        self.destroy_node()
        rclpy.shutdown()