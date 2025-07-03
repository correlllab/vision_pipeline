import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
import warnings
import matplotlib.pyplot as plt
import cv2
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

import numpy as np
import json
import random


import os

import sys


"""
Cheat Imports
"""
core_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(core_dir, "..")
utils_dir = os.path.join(parent_dir, "utils")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)


config_path = os.path.join(parent_dir, 'config.json')
config = json.load(open(config_path, 'r'))

fig_dir = os.path.join(parent_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(os.path.join(fig_dir, "SAM2"), exist_ok=True)



def get_points_and_colors(depths, rgbs, fx, fy, cx, cy):
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


#Class to use sam2
class SAM2_PC:
    def __init__(self,):
        """
        Initializes the SAM2 model and processor.
        Parameters:
        - iou_th: IoU threshold for NMS
        """
        self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else self.device

    def get_masks(self, rgb_img, depth_img, bbox, debug=False):
        #Run sam2 on all the boxes
        if debug:
            print(f"[get_masks] rgb_img.shape = {rgb_img.shape}")
            print(f"[get_masks] depth_img.shape = {depth_img.shape}")
            print(f"[get_masks] bbox = {bbox}, type = {type(bbox)}, np.array(bbox).shape = {np.array(bbox).shape}")
        if len(bbox) == 0:
            if debug:
                print("[get_masks] no boxes to process, returning empty masks")
            return  None, None
        self.sam_predictor.set_image(rgb_img.copy())
        sam_mask = None
        sam_scores = None
        sam_logits = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            original_sam_mask, sam_scores, sam_logits = self.sam_predictor.predict(box=bbox)
        if debug:
            print(f"[get_masks] original_sam_mask.shape = {original_sam_mask.shape}")
            print(f"[get_masks] sam_scores.shape = {sam_scores.shape}")
            print(f"[get_masks] sam_logits.shape = {sam_logits.shape}")
        if original_sam_mask.ndim == 3:
            # single mask → add batch axis
            original_sam_mask = original_sam_mask[np.newaxis, ...]
        sam_mask = np.all(original_sam_mask, axis=1)
        if debug:
            print(f"{sam_mask.shape=}")


        #Apply mask to the depth and rgb images
        #print(f"{original_sam_mask.shape=}, {sam_mask.shape=}, {rgb_img.shape=}, {depth_img.shape=}")
        masked_depth = depth_img[None, ...] * sam_mask
        masked_rgb = rgb_img[None, ...] * sam_mask[..., None]
        return masked_depth, masked_rgb

    def get_pcd_bbox(self, pts, cls):
        # build Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.numpy())
        pcd.colors = o3d.utility.Vector3dVector(cls.numpy()/255)
        pcd = pcd.voxel_down_sample(voxel_size=config["voxel_size"])
        # Apply statistical outlier removal to denoise the point cloud
        if config["statistical_outlier_removal"]:
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=config["statistical_nb_neighbors"], std_ratio=config["statistical_std_ratio"])
        if config["radius_outlier_removal"]:
            pcd, ind = pcd.remove_radius_outlier(nb_points=config["radius_nb_points"], radius=config["radius_radius"])
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox.color = (1.0, 0.0, 0.0)  # red

        return pcd, bbox


    def predict(self, rgb_img, depth_img, bbox, scores, intrinsics, debug = False, query_str=""):
        """
        Predicts 3D point clouds from RGB and depth images and bounding boxes using SAM2.
        Cleans up the point clouds and applies NMS.
        Parameters:
        - rgb_img: RGB image
        - depth_img: Depth image
        - bbox: Bounding boxes
        - intrinsics: Camera intrinsics
        - debug: If True, prints debug information
        Returns:
        - reduced_pcds: List of reduced point clouds
        - reduced_bounding_boxes_3d: List of reduced 3D bounding boxes
        - reduced_scores: List of reduced scores
        """
        if debug:
            print(f"[predict] Received {len(bbox)} boxes, scores shape = {scores.shape}")
            print(f"[predict] query_str = {query_str!r}")
        masked_depth, masked_rgb = self.get_masks(rgb_img, depth_img, bbox, debug=debug)
        if masked_depth is None or masked_rgb is None:
            return [], [], torch.tensor([]), [], []
        if debug:
            print(f"[predict] masked_depth.shape = {masked_depth.shape}")
            print(f"[predict] masked_rgb.shape   = {masked_rgb.shape}")
        tensor_depth = torch.from_numpy(masked_depth).to(self.device)
        tensor_rgb = torch.from_numpy(masked_rgb).to(self.device)

        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        points, colors = get_points_and_colors(tensor_depth, tensor_rgb, fx, fy, cx, cy)
        colors = colors[..., [2, 1, 0]]
        if debug:
            print(f"{points.shape=}, {colors.shape=}")

        B, N, _ = points.shape

        pcds = []
        bounding_boxes_3d = []
        new_masked_rgb = []
        new_masked_depth = []
        new_scores = []

        pts_cpu   = points.detach().cpu()
        cols_cpu  = colors.detach().cpu()
        #for each candiate object get the point cloud unless there are too few points
        for i in range(B):
            pts = pts_cpu[i]          # (N,3)
            cls = cols_cpu[i]         # (N,3)

            # mask out void points
            depths = pts[:, 2]
            valid = (depths > config["min_depth"])# & (depths < config["max_depth"])
            if debug:
                print(f"{valid.sum()=}")
            if valid.sum() <  config["min_3d_points"]:
                continue
            pts = pts[valid]
            cls = cls[valid]
            pcd, bbox = self.get_pcd_bbox(pts, cls)

            pcds.append(pcd)
            bounding_boxes_3d.append(bbox)
            new_masked_rgb.append(masked_rgb[i])
            new_masked_depth.append(masked_depth[i])
            new_scores.append(scores[i])

        new_scores = torch.tensor(new_scores)
        return pcds, bounding_boxes_3d, new_scores, masked_rgb, masked_depth
    def __str__(self):
        return f"SAM2: {self.sam_predictor.model.device}"
    def __repr__(self):
        return self.__str__()

def display_3dCandidates(predicitons, window_prefix = ""):
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer(window_prefix + "PointClouds", 1024, 768)
    vis.add_geometry("CameraFrame", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0]))

    for obj in predicitons:
        pcds = predicitons[obj]["pcds"]
        bboxes = predicitons[obj]["boxes"]
        scores = predicitons[obj]["scores"]

        for i, pcd in enumerate(pcds):
            vis.add_geometry(f"{obj}_pcd_{i}", pcd)
            vis.add_geometry(f"{obj}_bbox_{i}", bboxes[i])
            vis.add_3d_label(pcd.get_center(), f"{obj} {scores[i]:.2f}")
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()

if __name__ == "__main__":
    # Example usage
    sam2 = SAM2_PC()
    rgb_img = cv2.imread("./ExampleImages/MushroomRGB.jpeg")
    depth_img = cv2.imread("./ExampleImages/MushroomDepth.png")
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    center_x = rgb_img.shape[1] // 2
    center_y = rgb_img.shape[0] // 2
    intrinsics = {"fx": 500, "fy": 500, "cx": center_x, "cy": center_y}  # Example intrinsics

    from BBBackBones import OWLv2, display_2dCandidates
    owl_v2 = OWLv2()
    queries = ["mushroom"]
    owl_results= owl_v2.predict(rgb_img, queries, debug=False)
    display_2dCandidates(rgb_img, owl_results, window_prefix="OWL_")
    print(f"{len(owl_results['mushroom']['boxes'])=}, {len(owl_results['mushroom']['scores'])=}")

    pcds, bboxes, scores, masked_rgb, masked_depth = sam2.predict(rgb_img, depth_img, owl_results["mushroom"]["boxes"], owl_results["mushroom"]["scores"], intrinsics, debug=True)
    candidates_3d = {"mushroom": {"pcds": pcds, "boxes": bboxes, "scores": scores, "masked_rgb": masked_rgb, "masked_depth": masked_depth}}
    display_3dCandidates(candidates_3d, window_prefix="SAM2_")