import time
import rclpy
import itertools
import shutil
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
exp_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(exp_dir, "..")
utils_dir = os.path.join(parent_dir, "utils")
core_dir = os.path.join(parent_dir, "core")
fig_dir = os.path.join(parent_dir, 'figures', 'calibration')
shutil.rmtree(fig_dir, ignore_errors=True)
os.makedirs(fig_dir, exist_ok=True)
ros_dir = os.path.join(parent_dir, "ROS")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if ros_dir not in sys.path:
    sys.path.insert(0, ros_dir)
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)
if exp_dir not in sys.path:
    sys.path.insert(0, exp_dir)
from behaviors import MainNode
from RosRealsense import RealSenseSubscriber

def save_camera_info(camera_info, filepath):
    """
    Convert a ROS2 CameraInfo message into a NumPy .npz file.
    Stores K, D, R, P matrices and image size.
    """
    # Intrinsic matrix K (3x3)
    K = np.array(camera_info.k, dtype=np.float64).reshape(3, 3)

    # Distortion coefficients
    D = np.array(camera_info.d, dtype=np.float64)

    # Rectification matrix R (3x3)
    R = np.array(camera_info.r, dtype=np.float64).reshape(3, 3)

    # Projection matrix P (3x4)
    P = np.array(camera_info.p, dtype=np.float64).reshape(3, 4)

    # Save to .npz
    np.savez(
        filepath,
        width=camera_info.width,
        height=camera_info.height,
        distortion_model=camera_info.distortion_model,
        D=D,
        K=K,
        R=R,
        P=P,
        binning_x=camera_info.binning_x,
        binning_y=camera_info.binning_y,
        roi_x_offset=camera_info.roi.x_offset,
        roi_y_offset=camera_info.roi.y_offset,
        roi_height=camera_info.roi.height,
        roi_width=camera_info.roi.width,
        roi_do_rectify=camera_info.roi.do_rectify,
    )

def get_corners(rgb, pattern_size=(10,7)):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners


def nothing(x):
    pass

def main():
    rclpy.init()
    camera_node = RealSenseSubscriber("/realsense/left_hand")
    intrinsic_path = os.path.join("/ros2_ws/src/vision_pipeline/vision_pipeline/figures/calibration_set", "intrinsics.npz")
    intrinsics_made = os.path.exists(intrinsic_path)
    print("camera initialized")
    i = 0
    while True:

        # fetch camera frame
        rgb = None
        pose = None
        try_count = 0
        while rgb is None and try_count < 5:
            try_count += 1
            rgb, depth, info, pose = camera_node.get_data()
            if not intrinsics_made:
                if info is not None:
                    save_camera_info(info, intrinsic_path)
                    print(f"Saved intrinsics to {intrinsic_path}")
                    intrinsics_made = True
            time.sleep(0.05)

        if rgb is not None:
            h, w, _ = rgb.shape
            display_img = rgb.copy()
            cv2.putText(display_img, f"{pose=}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            success, corners = get_corners(display_img)
            if success:
                cv2.drawChessboardCorners(display_img, (10,7), corners, success)
            cv2.imshow("RGB", display_img)

        # quit on ESC
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord('s') and success and info is not None:
            stamp = f"{i=}"
            cv2.imwrite(os.path.join(fig_dir, f"calib_{stamp}_rgb.png"), rgb)
            np.savez(os.path.join(fig_dir, f"calib_{stamp}_corners.npz"), corners=corners, pose=pose)
            print(f"Saved calib_{stamp}_rgb.png and calib_{stamp}_corners.npz")
            i+=1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()