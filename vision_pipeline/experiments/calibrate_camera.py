import itertools
import time
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
from behaviors import BehaviorNode
from RosRealsense import RealSenseSubscriber
import threading
import random

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
ready_to_save = False
def vis_and_save(camera_node, intrinsic_path, intrinsics_made):
    i = 0
    global ready_to_save
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

        if rgb is not None and pose is not None:
            h, w, _ = rgb.shape
            display_img = rgb.copy()
            cv2.putText(display_img, f"{pose=}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

            success, corners = get_corners(rgb)
            if success:
                cv2.drawChessboardCorners(display_img, (10,7), corners, success)
                stamp = f"{i=}"
                lin_diff = 0
                ang_diff = 0

                
                # print(f"{lin_diff=}, {ang_diff=}")
                if ready_to_save:
                    cv2.imwrite(os.path.join(fig_dir, f"calib_{stamp}_rgb.png"), rgb)
                    cv2.imwrite(os.path.join(fig_dir, f"calib_{stamp}_corners.png"), display_img)
                    np.savez(os.path.join(fig_dir, f"calib_{stamp}_corners.npz"), corners=corners, pose=pose)
                    print(f"Saved calib_{stamp}_rgb.png and calib_{stamp}_corners.npz" )
                    i+=1
                    ready_to_save = False
            cv2.imshow("rgb", display_img)

        # quit on ESC
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            break
            
    cv2.destroyAllWindows()

def get_lookat_matrix(x,y,z,roll, target):
    pos = np.array([x, y, z], dtype=float)
    dir_vec = target - pos
    norm = np.linalg.norm(dir_vec)
    x_axis = dir_vec / norm

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, x_axis)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
        
    y0 = up - np.dot(up, x_axis) * x_axis
    y0 /= np.linalg.norm(y0)
    z0 = np.cross(x_axis, y0)
    z0 /= np.linalg.norm(z0)

    roll = np.deg2rad(roll)
    c, s = np.cos(roll), np.sin(roll)

    # rotate y0,z0 around x_axis by 'roll' (Rodrigues)
    # v_rot = v*c + (k×v)*s + k*(k·v)*(1-c), here k = x_axis
    k = x_axis
    y_axis = y0 * c + np.cross(k, y0) * s + k * np.dot(k, y0) * (1 - c)  # dot=0, so last term is 0
    z_axis = z0 * c + np.cross(k, z0) * s + k * np.dot(k, z0) * (1 - c)  # dot=0, so last term is 0


    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    return T

def main():
    behavior_node = BehaviorNode(vp = False)
    camera_node = RealSenseSubscriber("/realsense/left_hand")
    intrinsic_path = os.path.join(fig_dir, "intrinsics.npz")
    intrinsics_made = os.path.exists(intrinsic_path)
    vis_thread = threading.Thread(target=vis_and_save, args=(camera_node, intrinsic_path, intrinsics_made))
    vis_thread.start()
    time.sleep(1)
    print()
    print("camera initialized")


    target_location = [0.0, 1.25, 0.3]
    behavior_node.publish_marker(target_location[0], target_location[1], target_location[2])
    target = np.array(target_location, dtype=float)


    x = 0.2
    y = 0.3
    z = 0.5
    roll = 0
    T = get_lookat_matrix(x, y, z, roll, target)
    behavior_node.send_arm_goal(left_mat=T, duration=5)

    global ready_to_save
    xs = [-0.2, 0.0, 0.2]
    ys = [0.1, 0.25, 0.5]
    zs = [-0.3, 0.0, 0.3]
    rolls = [0, 45, 90, 135, 180, 270]
    configs = list(itertools.product(xs, ys, zs, rolls))
    random.shuffle(configs)
    for i, (x, y, z, roll) in enumerate(configs):
        saved = False
        while not saved:
            print(f"\n\n{i+1}/{len(configs)} New position: x={x}, y={y}, z={z}, roll={roll}")
            T = get_lookat_matrix(x, y, z, roll, target)
            behavior_node.send_arm_goal(left_mat=T, duration=5)
        
            cmd = input("Enter x y z r or dx dy dz dr or 'q' to quit, s to save, h for home, or k to skip: ")
            if cmd.strip().lower() in ['q', 'quit', 'exit']:
                break
            if cmd == "s":
                ready_to_save = True
                n_tries = 0
                while ready_to_save and n_tries < 5: #wait for other thread to set it back to False
                    time.sleep(0.1)
                    n_tries+=1
                saved = True
                continue
            if cmd == "k":
                saved = True
                continue
            if cmd == "h":
                behavior_node.go_home()
                continue
            value = input("Enter value: ")
            try:
                value = float(value)
            except ValueError:
                print("Invalid value. Please enter a numeric value.")
                continue
            if cmd.startswith('d'):
                if 'x' in cmd:
                    x += value
                if 'y' in cmd:
                    y += value
                if 'z' in cmd:
                    z += value
                if 'r' in cmd:
                    roll += value
            else:
                if 'x' in cmd:
                    x = value
                if 'y' in cmd:
                    y = value
                if 'z' in cmd:
                    z = value
                if 'r' in cmd:
                    roll = value
            

            
    

if __name__ == "__main__":
    main()