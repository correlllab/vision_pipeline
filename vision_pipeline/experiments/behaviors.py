import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from custom_ros_messages.srv import Query, UpdateTrackedObject
from custom_ros_messages.action import DualArm
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R
import time
from unitree_go.msg import MotorCmds, MotorCmd

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


class MainNode(Node):
    def __init__(self):
        rclpy.init()
        super().__init__('coordinator')
        # self.update_client = self.create_client(UpdateTrackedObject, 'vp_update_tracked_object')
        # self.query_client = self.create_client(Query, 'vp_query_tracked_objects')
        # while not self.update_client.wait_for_service(timeout_sec=1.0):
        #     print('Update Service not available, waiting again...')
        # while not self.query_client.wait_for_service(timeout_sec=1.0):
        #     print('Query Service not available, waiting again...')
        self.action_client = ActionClient(
            self,
            DualArm,
            'move_dual_arm'
        )
        self.hand_pub = self.create_publisher(MotorCmds, '/inspire/cmd', 10)



        self.r_arm_goal = None
        self.l_arm_goal = None
        self.l_hand = None
        self.r_hand = None
        self.go_home()
        
        

        self.open_hands()
        time.sleep(1)
        self.close_hands()
        time.sleep(1)

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
        r_hand = [0.0]*6
        self.set_hands(l_goal = l_hand, r_goal = r_hand)

    def go_home(self):
        
        res = self.send_arm_goal(
            left_arr = [0.3, 0.5, 0.2, 0, 0, 0],
            right_arr = [0.3, -0.5, 0.2, 0, 0, 0]
        )
        self.close_hands()
        time.sleep(1)


    def point_at(self, x,y,z):
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
        x_new = x
        y_new = y
        hand_length = 0.3 #30mm
        z_new = z + hand_length

        input("decend ? CTRL c to cancel")
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
        self.get_logger().info('Sending goal...')
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
        self.get_logger().info('Goal accepted, waiting for result...')


        # wait till finish
        future_result = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, future_result)
        result = future_result.result().result
        self.get_logger().info(f'Final result: success = {result.success}')
        print()
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        print(f'\rLeft Error Linear: {feedback.left_error_linear:.2f} Angular: {feedback.left_error_angular:.2f}; Right Error Linear: {feedback.right_error_linear:.2f} Angular: {feedback.right_error_linear:.2f} {time.time()- self.start_time:.2f}', end="", flush=True)

    def close(self):
        self.destroy_node()
        rclpy.shutdown()