from VisionPipeline import VisionPipe
from h12_controller.controller import ArmController
from RealsenseInterface import RealSenseCameraSubscriber
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from utils import pose_to_matrix, matrix_to_pose
import threading
if __name__ == "__main__":
    arm_controller = ArmController('/home/humanoid/Programs/H12Controller/assets/h1_2/h1_2.urdf',
                                   dt=0.01,
                                   vlim=1.0,
                                   visualize=True)
    vp = VisionPipe()

    head_sub = RealSenseCameraSubscriber(
        channel_name="realsense/Head",
        InitChannelFactory=False
    )
    larm_sub = RealSenseCameraSubscriber(
        channel_name="realsense/LArm",
        InitChannelFactory=False
    )
    # Initialize the arm controller

    arm_controller.left_ee_target_pose = [0.4, 0.2, 0.25, 0, 0.5, 0]
    def threaded_control():
        while True:
            #arm_controller.sim_dual_arm_step
            arm_controller.control_dual_arm_step()
            time.sleep(0.01)
    control_thread = threading.Thread(target=threaded_control)
    control_thread.start()

    objects = ["drill", "screwdriver", "wrench"]
    while True:
        #arm_controller.control_dual_arm_step()
        left_ee_pose = matrix_to_pose(arm_controller.left_ee_transformation)
        head_pose = matrix_to_pose(arm_controller.robot_model.get_frame_transformation('head_camera_link'))
        print(f"Left EE Pose: {left_ee_pose}")
        print(f"Head Pose: {head_pose}")

        """
        rgb_img, depth_img, Intrinsics, Extrinsics = None, None, None, None
        while rgb_img is None or depth_img is None or Intrinsics is None or Extrinsics is None:
            print("Waiting for RGB-D data...")
            rgb_img, depth_img, Intrinsics, Extrinsics = head_sub.read(display=False)

        I = {
            "fx": Intrinsics[0, 0],
            "fy": Intrinsics[1, 1],
            "cx": Intrinsics[0, 2],
            "cy": Intrinsics[1, 2],
            "width":rgb_img.shape[1],
            "height":rgb_img.shape[0],
        }
        print(f"{rgb_img.shape=}, {depth_img.shape=}")
        vp.update(rgb_img, depth_img, objects, I, head_pose)
        """


        rgb_img, depth_img, Intrinsics, Extrinsics = None, None, None, None
        while rgb_img is None or depth_img is None or Intrinsics is None or Extrinsics is None:
            print("Waiting for RGB-D data...")
            rgb_img, depth_img, Intrinsics, Extrinsics = larm_sub.read(display=False)

        I = {
            "fx": Intrinsics[0, 0],
            "fy": Intrinsics[1, 1],
            "cx": Intrinsics[0, 2],
            "cy": Intrinsics[1, 2],
            "width":rgb_img.shape[1],
            "height":rgb_img.shape[0],
        }
        vp.update(rgb_img, depth_img, objects, I, left_ee_pose)
        time.sleep(1)
    print("Robot system initialized and running.")