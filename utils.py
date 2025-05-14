import torch
import numpy as np
import open3d as o3d
def batch_backproject(depths, rgbs, fx, fy, cx, cy):
    """
    Back-project a batch of depth and RGB images to 3D point clouds.
    
    Args:
        depths: Tensor of shape (B, H, W) representing depth in meters.
        rgbs: Tensor of shape (B, H, W, 3) representing RGB colors, range [0, 1] or [0, 255].
        fx, fy, cx, cy: camera intrinsics.
    
    Returns:
        points: Tensor of shape (B, H*W, 3) representing 3D points.
        colors: Tensor of shape (B, H*W, 3) representing RGB colors for each point.
    """
    B, H, W = depths.shape
    device = depths.device
    
    # Create meshgrid of pixel coordinates
    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')  # shape (H, W)
    
    # Flatten pixel coordinates
    grid_u_flat = grid_u.reshape(-1)  # (H*W,)
    grid_v_flat = grid_v.reshape(-1)  # (H*W,)
    
    # Flatten depth and color
    z = depths.reshape(B, -1)  # (B, H*W)
    colors = rgbs.reshape(B, -1, 3)  # (B, H*W, 3)
    
    # Back-project to camera coordinates
    x = (grid_u_flat[None, :] - cx) * z / fx  # (B, H*W)
    y = (grid_v_flat[None, :] - cy) * z / fy  # (B, H*W)
    
    # Stack into point sets
    points = torch.stack((x, y, z), dim=-1)  # (B, H*W, 3)
    
    return points, colors

def iou_3d(bbox1: o3d.geometry.AxisAlignedBoundingBox, bbox2: o3d.geometry.AxisAlignedBoundingBox) -> float:
    """
    Compute the 3D Intersection over Union (IoU) of two Open3D axis-aligned bounding boxes.
    
    Args:
        bbox1: An open3d.geometry.AxisAlignedBoundingBox instance.
        bbox2: An open3d.geometry.AxisAlignedBoundingBox instance.
    
    Returns:
        IoU value as a float in [0.0, 1.0].
    """
    # Get the min and max corner coordinates of each box
    min1 = np.array(bbox1.get_min_bound(), dtype=np.float64)
    max1 = np.array(bbox1.get_max_bound(), dtype=np.float64)
    min2 = np.array(bbox2.get_min_bound(), dtype=np.float64)
    max2 = np.array(bbox2.get_max_bound(), dtype=np.float64)

    # Compute the intersection box bounds
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    # Compute intersection dimensions (clamp to zero if no overlap)
    inter_dims = np.clip(inter_max - inter_min, a_min=0.0, a_max=None)
    inter_vol = np.prod(inter_dims)

    # Compute volumes of each box
    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)

    # Compute union volume
    union_vol = vol1 + vol2 - inter_vol
    if union_vol <= 0:
        return 0.0

    return float(inter_vol / union_vol)