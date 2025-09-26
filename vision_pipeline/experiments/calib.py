import os
import glob
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rscipy

# --- Config ---
pattern_size = (10, 7)   # inner corners per row/col
square_size = 0.025      # meters
calib_folder = "/ros2_ws/src/vision_pipeline/vision_pipeline/figures/calibration_set"
intrinsics_file = os.path.join(calib_folder, "intrinsics.npz")


def create_object_points(pattern_size, square_size):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                           0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def rpy_to_matrix(roll, pitch, yaw, degrees=True):
    return Rscipy.from_euler('xyz', [roll, pitch, yaw], degrees=degrees).as_matrix()


def main():
    # Load intrinsics
    intrinsics = np.load(intrinsics_file)
    K = intrinsics["K"]
    dist = intrinsics["D"]

    # Load dataset
    dataset = []
    for fname in sorted(glob.glob(os.path.join(calib_folder, "*_corners.npz"))):
        data = np.load(fname)
        dataset.append((data["corners"], data["pose"], fname))
    print(f"Loaded {len(dataset)} samples")

    objp = create_object_points(pattern_size, square_size)

    R_gripper2base, t_gripper2base = [], []
    R_target2cam, t_target2cam = [], []

    prev_Rg, prev_tg = None, None
    prev_Rc, prev_tc = None, None

    for corners, pose, fname in dataset:
        if corners.shape[0] != objp.shape[0]:
            print(f"Skipping {fname} (got {corners.shape[0]} corners, expected {objp.shape[0]})")
            continue

        # --- Camera pose wrt target ---
        corners_2d = corners.reshape(-1, 2)
        ret, rvec, tvec = cv2.solvePnP(objp, corners_2d, K, dist)
        if not ret:
            print(f"SolvePnP failed for {fname}")
            continue
        R_cam, _ = cv2.Rodrigues(rvec)

        # --- Robot pose wrt base ---
        x, y, z, roll, pitch, yaw = pose
        Rg = rpy_to_matrix(roll, pitch, yaw, degrees=True)
        tg = np.array([[x], [y], [z]])

        # If we have a previous frame, compute relative motions
        if prev_Rg is not None and prev_Rc is not None:
            # Robot motion A
            R_A = prev_Rg.T @ Rg
            t_A = prev_Rg.T @ (tg - prev_tg)

            # Camera motion B
            R_B = prev_Rc.T @ R_cam
            t_B = prev_Rc.T @ (tvec - prev_tc)

            R_gripper2base.append(R_A)
            t_gripper2base.append(t_A)
            R_target2cam.append(R_B)
            t_target2cam.append(t_B)

        prev_Rg, prev_tg = Rg, tg
        prev_Rc, prev_tc = R_cam, tvec

    print(f"Collected {len(R_gripper2base)} motion pairs")

    # --- Hand-eye calibration ---
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    print("\n=== Hand–Eye Calibration Result ===")
    print("Rotation (R):\n", R_cam2gripper)
    print("Translation (t):\n", t_cam2gripper.ravel())

    # Save to file
    out_file = os.path.join(calib_folder, "handeye_result.npz")
    np.savez(out_file, R=R_cam2gripper, t=t_cam2gripper)
    print(f"Saved hand–eye result to {out_file}")


if __name__ == "__main__":
    main()
