#!/usr/bin/env python3
"""
Eye-in-Hand calibration (hardcoded paths & settings) + per-method reprojection stats

- Reads chessboard corners + robot poses from:
    /ros2_ws/src/vision_pipeline/vision_pipeline/figures/calibration
    (expects many *.npz, each containing keys: 'corners' and 'pose')

- Reads camera intrinsics from:
    /ros2_ws/src/vision_pipeline/vision_pipeline/figures/calibration/intrinsics.npz
    (expects keys: K, D, width, height, distortion_model, R, P — per your save_camera_info)

- Chessboard:
    inner corners = 10 x 7   (11x8 squares)
    square size   = 0.02 m   (2 cm)

- Tries every OpenCV hand-eye method: TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS
  and prints the resulting ^gT_c (camera in gripper frame) for each, plus
  per-method reprojection RMSE stats (mean / median / max) over all kept views.

Requires: OpenCV >= 4.1
"""

import os
import glob
from typing import List, Tuple
import numpy as np
import cv2

# -------------------- HARDCODED CONFIG --------------------
DATA_DIR = "/ros2_ws/src/vision_pipeline/vision_pipeline/figures/calibration"
assert os.path.exists(DATA_DIR), f"Data dir not found: {DATA_DIR}"
INTRINSICS_PATH = "/ros2_ws/src/vision_pipeline/vision_pipeline/figures/calibration/intrinsics.npz"
assert os.path.exists(INTRINSICS_PATH), f"Intrinsics file not found: {INTRINSICS_PATH}"

INNER_CORNERS = (10, 7)      # (cols, rows)
SQUARE_SIZE_M = 0.02         # 2cm
REPROJ_RMSE_THRESH_PX = 1.5  # drop bad views above this reprojection error

HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}
# -----------------------------------------------------------


# ----------------------------- Utils -----------------------------
def quat_to_R(q: np.ndarray, wxyz: bool = True) -> np.ndarray:
    q = np.asarray(q, dtype=float).ravel()
    if not wxyz:  # [x,y,z,w] -> [w,x,y,z]
        q = np.array([q[3], q[0], q[1], q[2]], dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero-length quaternion")
    w, x, y, z = q / n
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R


def Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def pose_to_Rt(pose_obj) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse 'pose' saved in the NPZ:
      - 4x4 homogeneous
      - [x,y,z,qx,qy,qz,qw] or [x,y,z,qw,qx,qy,qz]
    """
    arr = np.asarray(pose_obj)
    if arr.shape == (4, 4):
        return arr[:3, :3].astype(float), arr[:3, 3].astype(float)
    arr = arr.ravel().astype(float)
    if arr.size == 7:
        p = arr[:3]
        q4 = arr[3:]
        try:
            R = quat_to_R([q4[3], q4[0], q4[1], q4[2]], wxyz=True)  # treat as qx,qy,qz,qw
        except Exception:
            R = quat_to_R(q4, wxyz=True)  # treat as qw,qx,qy,qz
        return R, p
    if arr.size == 16:
        return pose_to_Rt(arr.reshape(4, 4))
    raise ValueError(f"Unsupported pose rep shape={arr.shape}, size={arr.size}")


# ----------------------- Data-specific loaders -----------------------
def load_intrinsics_npz(path: str):
    d = np.load(path, allow_pickle=True)
    K = d["K"].astype(float)
    D = d["D"].astype(float).ravel()
    distortion_model = str(d["distortion_model"]) if "distortion_model" in d else ""
    width = int(d["width"]) if "width" in d else None
    height = int(d["height"]) if "height" in d else None
    R = d["R"].astype(float) if "R" in d else None
    P = d["P"].astype(float) if "P" in d else None
    return K, D, distortion_model, width, height, R, P


def make_chessboard_object_points(inner_corners, square_size_m):
    cols, rows = inner_corners
    objp = np.zeros((rows * cols, 3), dtype=np.float64)
    gx, gy = np.meshgrid(np.arange(cols), np.arange(rows))
    objp[:, :2] = np.vstack([gx.ravel(), gy.ravel()]).T * square_size_m
    return objp


def solve_pnp_for_view(image_points: np.ndarray,
                       obj_points: np.ndarray,
                       K: np.ndarray,
                       D: np.ndarray,
                       distortion_model: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (R_target2cam, t_target2cam, reproj_rmse_px).
    If fisheye/equidistant, undistort points first (cv2.fisheye) and solvePnP with zero distortion.
    """
    img_pts = np.asarray(image_points, dtype=np.float64)
    if img_pts.ndim == 3 and img_pts.shape[1:] == (1, 2):
        img_pts = img_pts.reshape(-1, 2)
    elif img_pts.ndim == 2 and img_pts.shape[1] == 2:
        pass
    else:
        raise ValueError(f"Unexpected corners shape {image_points.shape}")

    obj_pts = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)
    distort = (distortion_model or "").lower()

    if "equidistant" in distort or "fisheye" in distort:
        pts = img_pts.reshape(-1, 1, 2)
        und = cv2.fisheye.undistortPoints(pts, K, D, P=K).reshape(-1, 2)
        use_K, use_D = K, None
        img_for_err = und
    else:
        und = img_pts
        use_K, use_D = K, D
        img_for_err = img_pts

    ok, rvec, tvec = cv2.solvePnP(obj_pts, und, use_K, use_D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, use_K, (np.zeros(5) if use_D is None else use_D))
    proj = proj.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((img_for_err - proj) ** 2, axis=1))))

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    return R.astype(float), t.astype(float), rmse


def ax_xb_residuals(
    R_g2b: List[np.ndarray], t_g2b: List[np.ndarray],
    R_t2c: List[np.ndarray], t_t2c: List[np.ndarray],
    R_g2c: np.ndarray, t_g2c: np.ndarray
) -> Tuple[float, float]:
    """Average residuals of A X = X B over all i<j pairs."""
    def to_T(R, t): 
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = t.ravel(); return T
    def invT(T): 
        R = T[:3,:3]; t = T[:3,3]; Ti = np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]=-R.T@t; return Ti

    Tg = [to_T(R, t) for R, t in zip(R_g2b, t_g2b)]   # ^bT_g
    Tt = [to_T(R, t) for R, t in zip(R_t2c, t_t2c)]   # ^cT_t
    X = to_T(R_g2c, t_g2c)

    r_err, t_err = [], []
    n = len(Tg)
    for i in range(n):
        for j in range(i+1, n):
            A = invT(Tg[j]) @ Tg[i]
            B = Tt[j] @ invT(Tt[i])
            dT = invT(A @ X) @ (X @ B)
            dR, dt = dT[:3, :3], dT[:3, 3]
            r_err.append(np.linalg.norm(dR - np.eye(3)))
            t_err.append(np.linalg.norm(dt))
    if not r_err:
        return float("nan"), float("nan")
    return float(np.mean(r_err)), float(np.mean(t_err))


def project_board_points(obj_pts_3d: np.ndarray,
                         R_c_t: np.ndarray,
                         t_c_t: np.ndarray,
                         K: np.ndarray,
                         D: np.ndarray,
                         distortion_model: str) -> np.ndarray:
    """Project 3D board points into pixels given ^cR_t, ^ct_t, handling fisheye if needed."""
    rvec, _ = cv2.Rodrigues(R_c_t)
    if "equidistant" in (distortion_model or "").lower() or "fisheye" in (distortion_model or "").lower():
        # Use fisheye projection
        # cv2.fisheye.projectPoints expects (N,1,3) object points and rvec/tvec (3,1)
        obj = obj_pts_3d.reshape(-1, 1, 3).astype(np.float64)
        tvec = t_c_t.reshape(3, 1).astype(np.float64)
        img_pts, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
        return img_pts.reshape(-1, 2)
    else:
        img_pts, _ = cv2.projectPoints(obj_pts_3d, rvec, t_c_t.reshape(3, 1), K, D)
        return img_pts.reshape(-1, 2)


# ------------------------------ Main routine ------------------------------
def main():
    # Load intrinsics
    K, D, distortion_model, width, height, R_rect, P = load_intrinsics_npz(INTRINSICS_PATH)
    print("[INFO] Intrinsics loaded:")
    print("K=\n", K)
    print("D=", D)
    print("distortion_model=", distortion_model)
    if width and height:
        print(f"image size: {width} x {height}")

    # Prepare chessboard object points
    obj_pts = make_chessboard_object_points(INNER_CORNERS, SQUARE_SIZE_M)

    # Gather samples
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {DATA_DIR}")

    R_t2c_list: List[np.ndarray] = []
    t_t2c_list: List[np.ndarray] = []
    R_g2b_list: List[np.ndarray] = []
    t_g2b_list: List[np.ndarray] = []
    reproj_errors: List[float] = []
    corners_list: List[np.ndarray] = []
    filenames: List[str] = []

    print("\n[STEP] Solving PnP for each view (and filtering by reprojection RMSE)...")
    for f in npz_files:
        with np.load(f, allow_pickle=True) as d:
            if "corners" not in d or "pose" not in d:
                continue
            corners = d["corners"]
            pose = d["pose"]

        try:
            Rgb, tgb = pose_to_Rt(pose)  # ^bR_g, ^bt_g
            Rc_t, tc_t, rmse = solve_pnp_for_view(corners, obj_pts, K, D, distortion_model)
        except Exception as e:
            print(f"[WARN] {os.path.basename(f)} skipped: {e}")
            continue

        if np.isfinite(rmse) and rmse <= REPROJ_RMSE_THRESH_PX:
            R_t2c_list.append(Rc_t)
            t_t2c_list.append(tc_t.reshape(3, 1))
            R_g2b_list.append(Rgb)
            t_g2b_list.append(tgb.reshape(3, 1))
            corners_list.append(corners.reshape(-1, 2).astype(np.float64))
            filenames.append(os.path.basename(f))
            reproj_errors.append(rmse)
            print(f"[OK]   {os.path.basename(f)}  RMSE={rmse:.3f}px")
        else:
            print(f"[DROP] {os.path.basename(f)}  RMSE={rmse:.3f}px > {REPROJ_RMSE_THRESH_PX:.2f}px")

    if len(R_t2c_list) < 3:
        raise RuntimeError("Not enough valid views after filtering (need >= 3).")

    mean_rmse = float(np.mean(reproj_errors)) if reproj_errors else float("nan")
    print(f"\n[INFO] Kept {len(reproj_errors)} views; mean PnP RMSE = {mean_rmse:.3f}px")

    # Precompute convenience transforms for each kept view
    T_b_g_list = [Rt_to_T(R, t) for R, t in zip(R_g2b_list, t_g2b_list)]
    T_c_t_list = [Rt_to_T(R, t) for R, t in zip(R_t2c_list, t_t2c_list)]  # ^cT_t per view

    # Try every hand-eye method
    print("\n================ Hand-Eye Results (camera-in-gripper, ^gT_c) ================\n")
    for name, flag in HAND_EYE_METHODS.items():
        try:
            R_g2c, t_g2c = cv2.calibrateHandEye(
                R_g2b_list, t_g2b_list,
                R_t2c_list, t_t2c_list,
                method=flag
            )
            R_g2c = np.asarray(R_g2c, float)
            t_g2c = np.asarray(t_g2c, float).reshape(3)
            T_g2c = Rt_to_T(R_g2c, t_g2c)

            # Diagnostics: AX=XB residuals
            rres, tres = ax_xb_residuals(
                R_g2b_list, t_g2b_list,
                R_t2c_list, t_t2c_list,
                R_g2c, t_g2c
            )

            # ---------- Per-method reprojection validation ----------
            # Use the first kept view to define board pose in base: ^bT_t = ^bT_g * ^gT_c * ^cT_t
            ref_idx = 0
            T_b_t = T_b_g_list[ref_idx] @ T_g2c @ T_c_t_list[ref_idx]

            per_view_rmse = []
            for i in range(len(T_b_g_list)):
                # ^bT_c(i) = ^bT_g(i) * ^gT_c  ->  ^cT_b(i) = inv(^bT_c(i))
                T_b_c_i = T_b_g_list[i] @ T_g2c
                T_c_b_i = invert_T(T_b_c_i)

                # Predicted ^cT_t for this view
                T_c_t_pred = T_c_b_i @ T_b_t
                R_c_t_pred = T_c_t_pred[:3, :3]
                t_c_t_pred = T_c_t_pred[:3, 3]

                # Project model, compare to detected corners
                proj = project_board_points(obj_pts, R_c_t_pred, t_c_t_pred, K, D, distortion_model)
                err = corners_list[i] - proj
                rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
                per_view_rmse.append(rmse)

            per_view_rmse = np.array(per_view_rmse, dtype=float)
            mean_e = float(np.mean(per_view_rmse))
            med_e  = float(np.median(per_view_rmse))
            max_e  = float(np.max(per_view_rmse))

            print(f"[METHOD] {name}")
            print("T_g2c =")
            print(T_g2c)
            print(f"diag: AX=XB residuals -> rot Frobenius={rres:.5f}, trans(m)={tres:.5f}")
            print(f"reproj (px): mean={mean_e:.3f}, median={med_e:.3f}, max={max_e:.3f}\n")

        except Exception as e:
            print(f"[METHOD] {name} failed: {e}\n")

    print("=============================== Done =================================")


if __name__ == "__main__":
    main()
