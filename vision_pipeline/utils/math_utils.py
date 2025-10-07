import numpy as np
import cv2
import open3d as o3d
import os
util_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(util_dir, "..")
if parent_dir not in os.sys.path:
    os.sys.path.insert(0, parent_dir)
from config import config  # global config dictionary

def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """
    Convert a 6-vector [x, y, z, roll, pitch, yaw] (radians)
    into a 4×4 homogeneous transform.
    """
    pose = [float(p) for p in pose]  # ensure float type
    x, y, z, roll, pitch, yaw = pose

    # Rotation about X axis (roll)
    Rx = np.array([
        [1,            0,             0],
        [0,  np.cos(roll), -np.sin(roll)],
        [0,  np.sin(roll),  np.cos(roll)],
    ])

    # Rotation about Y axis (pitch)
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])

    # Rotation about Z axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1],
    ])

    # Combined rotation: R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T

def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """
    Convert a 4×4 homogeneous matrix back into a 6-vector
    [x, y, z, roll, pitch, yaw] with angles in radians.
    Assumes T[:3,:3] = Rz @ Ry @ Rx.
    """
    # translation
    x, y, z = T[:3, 3]

    # rotation matrix
    R = T[:3, :3]

    # recover pitch = asin(–R[2,0])
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))

    # to avoid gimbal‐lock edge cases you could test cos(pitch)≈0
    # but for most cases:
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw  = np.arctan2(R[1, 0], R[0, 0])

    return np.array([x, y, z, roll, pitch, yaw])

def quat_to_euler(x, y, z, w):
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return np.array([roll, pitch, yaw])

def in_image(
    pcd: o3d.t.geometry.PointCloud,
    obs_transform: np.ndarray,
    I: dict,
) -> bool:
    """
    Return True if at least `visible_portion_requirement` of the point cloud
    projects inside the image bounds with z_cam > 0.
    """
    # --- read & validate threshold (global config) ---
    thr = float(config['visible_portion_requirement'])
    if not (0.0 < thr < 1.0):
        raise ValueError("config['visible_portion_requirement'] must be in (0,1).")

    # --- get point positions as numpy (CPU) ---
    if len(pcd.point) == 0:
        return False

    if "positions" in pcd.point:
        pts = pcd.point["positions"].cpu().numpy()  # (N,3)
    else:
        try:
            pts = pcd.point.positions.cpu().numpy()
        except AttributeError:
            raise ValueError("PointCloud has no 'positions' attribute.")

    N = pts.shape[0]
    if N == 0:
        return False

    # --- world→camera transform ---
    T_cam2world = obs_transform
    T_world2cam = np.linalg.inv(T_cam2world)

    # Vectorized homogeneous transform
    pts_h = np.concatenate([pts, np.ones((N, 1), dtype=pts.dtype)], axis=1)
    p_cam = pts_h @ T_world2cam.T
    x_cam, y_cam, z_cam = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]

    # In front of camera
    front = z_cam > 0.0
    if not np.any(front):
        return False

    # --- pinhole projection ---
    u = I['fx'] * (x_cam[front] / z_cam[front]) + I['cx']
    v = I['fy'] * (y_cam[front] / z_cam[front]) + I['cy']

    # --- bounds check ---
    in_x = (u >= 0.0) & (u < float(I['width']))
    in_y = (v >= 0.0) & (v < float(I['height']))
    visible_mask = in_x & in_y

    visible_fraction = visible_mask.sum() / float(N)
    return visible_fraction >= thr

def is_obscured(
    pcd: o3d.t.geometry.PointCloud,
    depth_image: np.ndarray,
    cam_transform,
    I: dict,
) -> bool:
    """
    Returns True if ≥ (1 - visible_portion_requirement) of the *valid projected* points
    are occluded by the depth image (depth closer than the point by > occlusion_tol).

    Notes:
      - Uses global `config["visible_portion_requirement"]` in (0,1).
      - Points behind the camera, outside bounds, or mapping to invalid depth
        samples are excluded from the denominator. If none remain, returns False.
    """
    # --- read threshold from global config and map to occluded fraction ---
    occlusion_tol = float(config['occlusion_tol'])
    thr = float(config['visible_portion_requirement'])
    if not (0.0 < thr < 1.0):
        raise ValueError("config['visible_portion_requirement'] must be in (0,1).")
    thr_occ = 1.0 - thr

    # --- fetch points (CPU numpy) ---
    if "positions" in pcd.point:
        pts = pcd.point["positions"].cpu().numpy()
    else:
        try:
            pts = pcd.point.positions.cpu().numpy()
        except AttributeError:
            raise ValueError("PointCloud has no 'positions' attribute.")
    if pts.shape[0] == 0:
        return False

    # --- world→camera transform ---
    T_wc = cam_transform
    T_cw = np.linalg.inv(T_wc)

    # --- transform to camera frame (vectorized) ---
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    pts_cam = (T_cw @ pts_h.T).T[:, :3]

    # --- keep points in front of camera ---
    z = pts_cam[:, 2]
    front = z > 0.0
    if not np.any(front):
        return False
    pts_cam = pts_cam[front]
    z = z[front]

    # --- project to pixel indices (nearest) ---
    fx, fy, cx, cy = I["fx"], I["fy"], I["cx"], I["cy"]
    u = np.round(pts_cam[:, 0] * fx / z + cx).astype(int)
    v = np.round(pts_cam[:, 1] * fy / z + cy).astype(int)

    H, W = int(I["height"]), int(I["width"])
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_bounds):
        return False

    u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]

    # --- sample depth and keep valid ---
    depth_samples = depth_image[v, u]
    valid_depth = (depth_samples > 0) & np.isfinite(depth_samples)
    if not np.any(valid_depth):
        return False

    z = z[valid_depth]
    depth_samples = depth_samples[valid_depth]

    # --- occlusion test ---
    occluded = depth_samples + occlusion_tol < z
    occ_frac = np.count_nonzero(occluded) / occluded.size

    return occ_frac >= thr_occ

def display_2dCandidates(img, predictions, window_prefix = "", display=False, save_path=None):
    """
    Displays the bounding box predictions on the image.
    Parameters:
    - img: The input image (numpy array).
    - predictions: Dictionary containing bounding box predictions. like
        {
            "query_object_1": {
                "boxes": [[x1, y1, x2, y2], ...],
                "probs": [prob1, prob2, ...]
            },
            "query_object_2": {
                "boxes": [[x1, y1, x2, y2], ...],
                "probs": [prob1, prob2, ...]
            },
            ...
            "query_object_N": {
                "boxes": [[x1, y1, x2, y2], ...],
                "probs": [prob1, prob2, ...]
            }
        }
    """
    display_img = img.copy()
    for query_object, prediction in predictions.items():
        for bbox, prob in zip(prediction["boxes"], prediction["probs"]):
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(display_img, f"{query_object} {prob:.4f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if display:
        cv2.imshow(f"{window_prefix}", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save_path is not None:
        cv2.imwrite(save_path, display_img)
    return display_img

def mean_nn_dist(source: o3d.t.geometry.PointCloud,
                 target: o3d.t.geometry.PointCloud) -> float:
    """
    Compute the mean nearest-neighbor distance from each point in `source`
    to the closest point in `target`.
    """
    assert not source.is_empty(), "source is empty"
    assert not target.is_empty(), "target is empty"

    # Create NN search index on the target points
    nns = o3d.core.nns.NearestNeighborSearch(target.point["positions"])
    nns.knn_index()

    # Query the nearest neighbor (k=1) for each source point
    indices, distances = nns.knn_search(source.point["positions"], 1)

    # distances is squared Euclidean distance (tensor shape [N,1])
    distances = distances.sqrt()

    # Return mean distance as Python float
    return float(distances.mean().item())

def _positions_np(pcd: o3d.t.geometry.PointCloud) -> np.ndarray:
    """Return (N,3) float64 NumPy array of positions from a tensor point cloud."""
    if pcd.is_empty():
        return np.empty((0, 3), dtype=np.float64)
    # Prefer standard key; fall back to attribute access for older/newer APIs
    if "positions" in pcd.point:
        t = pcd.point["positions"]
    else:
        try:
            t = pcd.point.positions
        except AttributeError:
            raise ValueError("PointCloud has no 'positions' attribute.")
    arr = t.cpu().numpy()
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"'positions' must have shape (N,3), got {arr.shape}")
    return arr.astype(np.float64, copy=False)

def _mean_cov(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (mean(3,), cov(3,3), n). Uses sample covariance (ddof=1)."""
    n = int(pts.shape[0])
    if n == 0:
        return np.zeros(3, dtype=np.float64), np.zeros((3,3), dtype=np.float64), 0
    mu = pts.mean(axis=0)
    if n < 2:
        return mu, np.zeros((3,3), dtype=np.float64), n
    C = np.cov(pts, rowvar=False, ddof=1)
    return mu, np.asarray(C, dtype=np.float64).reshape(3,3), n

def mahalanobis_distance(
    pcd_a: o3d.t.geometry.PointCloud,
    pcd_b: o3d.t.geometry.PointCloud,
    cov_mode: str = "pooled",   # {"pooled","avg","a","b","diag-pooled","identity"}
    eps: float = 1e-6,
) -> float:
    """
    Mahalanobis distance between two point clouds modeled as Gaussians:
        d = sqrt( (μa - μb)^T Σ^{-1} (μa - μb) )
    Σ is chosen by `cov_mode`. Uses only NumPy and the tensor API.
    """
    A = _positions_np(pcd_a)
    B = _positions_np(pcd_b)
    if A.shape[0] == 0 or B.shape[0] == 0:
        raise ValueError("Both point clouds must contain at least one point.")

    mu_a, C_a, n_a = _mean_cov(A)
    mu_b, C_b, n_b = _mean_cov(B)
    dmu = mu_a - mu_b

    cov_mode = cov_mode.lower()
    if cov_mode == "pooled":
        denom = max(n_a + n_b - 2, 1)
        if n_a >= 2 and n_b >= 2:
            Sigma = ((max(n_a-1,0)*C_a) + (max(n_b-1,0)*C_b)) / denom
        elif n_a >= 2:
            Sigma = C_a
        elif n_b >= 2:
            Sigma = C_b
        else:
            Sigma = np.eye(3)
    elif cov_mode == "avg":
        Sigma = 0.5 * (C_a + C_b)
    elif cov_mode == "a":
        Sigma = C_a if n_a >= 2 else (C_b if n_b >= 2 else np.eye(3))
    elif cov_mode == "b":
        Sigma = C_b if n_b >= 2 else (C_a if n_a >= 2 else np.eye(3))
    elif cov_mode == "diag-pooled":
        # diagonal-only pooled covariance for robustness
        denom = max(n_a + n_b - 2, 1)
        if n_a >= 2 and n_b >= 2:
            pooled = ((max(n_a-1,0)*C_a) + (max(n_b-1,0)*C_b)) / denom
        elif n_a >= 2:
            pooled = C_a
        elif n_b >= 2:
            pooled = C_b
        else:
            pooled = np.eye(3)
        Sigma = np.diag(np.clip(np.diag(pooled), eps, None))
    elif cov_mode == "identity":
        Sigma = np.eye(3)
    else:
        raise ValueError(f"Unknown cov_mode '{cov_mode}'.")

    # Regularize and compute distance (prefer Cholesky; fallback to pinv)
    if not np.isfinite(Sigma).all() or np.allclose(Sigma, 0):
        Sigma = np.eye(3)
    Sigma_reg = Sigma + float(eps) * np.eye(3)

    try:
        L = np.linalg.cholesky(Sigma_reg)
        y = np.linalg.solve(L, dmu)
        dist2 = float(y @ y)
    except np.linalg.LinAlgError:
        dist2 = float(dmu @ (np.linalg.pinv(Sigma_reg) @ dmu))

    return float(np.sqrt(max(dist2, 0.0)))
