from VisionPipeline import VisionPipe
from h12_controller.controller import ArmController
from RealsenseInterface import RealSenseCameraSubscriber
import time
if __name__ == "__main__":

    # Initialize the arm controller
    arm_controller = ArmController('/home/max/programs/H12Controller/assets/h1_2/h1_2.urdf',
                                   dt=0.01,
                                   vlim=1.0,
                                   visualize=True)
    arm_controller.left_ee_target_pose = [1, 1, 1, 0, 0, 0]
    while True:
        #arm_controller.sim_dual_arm_step()
        arm_controller.control_dual_arm_step()
        print(arm_controller.left_ee_transformation)
        print(arm_controller.robot_model.get_frame_transformation('head_camera_link'))
        time.sleep(0.01)
    print("Robot system initialized and running.")