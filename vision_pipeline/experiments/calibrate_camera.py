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


def vis_and_save(camera_node, intrinsic_path, intrinsics_made):
    i = 0
    save_cooldown = 5.0
    last_save = time.time() - 10
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

            success, corners = get_corners(rgb)
            if success:
                cv2.drawChessboardCorners(display_img, (10,7), corners, success)
                stamp = f"{i=}"
                if time.time() - last_save > save_cooldown:
                    cv2.imwrite(os.path.join(fig_dir, f"calib_{stamp}_rgb.png"), rgb)
                    cv2.imwrite(os.path.join(fig_dir, f"calib_{stamp}_corners.png"), display_img)
                    np.savez(os.path.join(fig_dir, f"calib_{stamp}_corners.npz"), corners=corners, pose=pose)
                    print(f"Saved calib_{stamp}_rgb.png and calib_{stamp}_corners.npz")
                    i+=1
                    last_save = time.time()
            cv2.imshow("RGB", display_img)

        # quit on ESC
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            break
            
    cv2.destroyAllWindows()


import numpy as np
import matplotlib.pyplot as plt

def _set_axes_equal(ax):
    # Make 3D axes have equal scale so triads aren't distorted
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_mid = np.mean(x_limits); y_mid = np.mean(y_limits); z_mid = np.mean(z_limits)
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

def plot_se3_frames(Ts, axis_len=0.1, stride=1, show_target=None, show=True, save_path=None):
    """
    Ts: list of 4x4 homogeneous transforms (row-major)
    axis_len: length of each triad axis
    stride: plot every Nth transform to reduce clutter
    show_target: optional (x,y,z) to plot a target point
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Collect positions for bounds
    Ps = []
    for i, T in enumerate(Ts[::max(1, stride)]):
        T = np.asarray(T, dtype=float).reshape(4,4)
        p = T[:3, 3]
        R = T[:3, :3]
        Ps.append(p)

        # triad directions
        x_dir, y_dir, z_dir = R[:, 0], R[:, 1], R[:, 2]

        # draw axis triad
        ax.quiver(p[0], p[1], p[2], *(x_dir*axis_len), linewidth=1, color='r')
        ax.quiver(p[0], p[1], p[2], *(y_dir*axis_len), linewidth=1, color='g')
        ax.quiver(p[0], p[1], p[2], *(z_dir*axis_len), linewidth=1, color='b')

    Ps = np.array(Ps) if Ps else np.zeros((1,3))
    ax.scatter(Ps[:,0], Ps[:,1], Ps[:,2], s=5)

    if show_target is not None:
        tx, ty, tz = show_target
        ax.scatter([tx], [ty], [tz], s=40, marker='*')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Transforms')
    # set bounds with a small margin
    mins = Ps.min(axis=0) - axis_len*2
    maxs = Ps.max(axis=0) + axis_len*2
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(min(mins[1], 0), maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    _set_axes_equal(ax)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax




def main():
    behavior_node = BehaviorNode(vp = False)
    camera_node = RealSenseSubscriber("/realsense/left_hand")
    intrinsic_path = os.path.join(fig_dir, "intrinsics.npz")
    intrinsics_made = os.path.exists(intrinsic_path)
    print("camera initialized")
    i = 0

    vis_thread = threading.Thread(target=vis_and_save, args=(camera_node, intrinsic_path, intrinsics_made))
    vis_thread.start()


    x_range = np.linspace(-0.1, 0.3, 2)
    y_range = np.linspace(-0.3, 0.1, 2)
    z_range = np.linspace(0.7, 0.8, 2)
    roll_range = np.linspace(-np.pi, np.pi, 1, endpoint=False)
    target_location = [0.45, 0.1, -0.15]
    behavior_node.publish_marker(target_location[0], target_location[1], target_location[2])
    target = np.array(target_location, dtype=float)
    t = np.eye(4)
    t[:3, 3] = target_location
    t[2, 3] += 0.1
    behavior_node.send_arm_goal(left_mat=t, duration=5)
    input("Press Enter to continue...")
    Ts = []
    for x, y, z in itertools.product(x_range, y_range, z_range):
        # behavior_node.go_home()
        # Compute rotation so x-axis points toward target_location
        pos = np.array([x, y, z], dtype=float) + target
        print(f"Moving to {pos}")
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


        for roll in roll_range:
            c, s = np.cos(roll), np.sin(roll)

        
            # rotate y0,z0 around x_axis by 'roll' (Rodrigues)
            # v_rot = v*c + (k×v)*s + k*(k·v)*(1-c), here k = x_axis
            k = x_axis
            y_axis = y0 * c + np.cross(k, y0) * s + k * np.dot(k, y0) * (1 - c)  # dot=0, so last term is 0
            z_axis = z0 * c + np.cross(k, z0) * s + k * np.dot(k, z0) * (1 - c)  # dot=0, so last term is 0


            T = np.eye(4)
            T[:3, 3] = pos
            T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
            Ts.append(T)
            behavior_node.send_arm_goal(left_mat=T, duration=5)
            time.sleep(1)
    

    plot_se3_frames(Ts, show_target=target_location)


if __name__ == "__main__":
    main()