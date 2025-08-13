import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
import warnings
import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.t.geometry as o3tg

import numpy as np
import json

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

from math_utils import pose_to_matrix

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
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')  # (H, W)

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
    def __init__(self):
        """
        Initializes the SAM2 model and processor.
        Parameters:
        - iou_th: IoU threshold for NMS
        """
        self.sam_predictor = SAM2ImagePredictor.from_pretrained(config["sam2_model"])
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.o3dDevice = o3c.Device("CPU:0") #o3c.Device("CUDA:0")

    def get_masks(self, rgb_img, depth_img, bbox, debug):
        #Run sam2 on all the boxes
        if debug:
            print(f"[SAM2_PC get_masks] rgb_img.shape = {rgb_img.shape}")
            print(f"[SAM2_PC get_masks] depth_img.shape = {depth_img.shape}")
            print(f"[SAM2_PC get_masks] bbox = {bbox}, type = {type(bbox)}, np.array(bbox).shape = {np.array(bbox).shape}")
        if len(bbox) == 0:
            if debug:
                print("[SAM2_PC get_masks] no boxes to process, returning empty masks")
            return  None, None
        self.sam_predictor.set_image(rgb_img.copy())
        sam_mask = None
        sam_scores = None
        sam_logits = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            original_sam_mask, sam_scores, sam_logits = self.sam_predictor.predict(box=bbox)
        if debug:
            print(f"[SAM2_PC get_masks] original_sam_mask.shape = {original_sam_mask.shape}")
            print(f"[SAM2_PC get_masks] sam_scores.shape = {sam_scores.shape}")
            print(f"[SAM2_PC get_masks] sam_logits.shape = {sam_logits.shape}")
        if original_sam_mask.ndim == 3:
            # single mask → add batch axis
            original_sam_mask = original_sam_mask[np.newaxis, ...]
        sam_mask = np.all(original_sam_mask, axis=1)
        if debug:
            print(f"[SAM2_PC get_masks] {sam_mask.shape=}")


        #Apply mask to the depth and rgb images
        #print(f"{original_sam_mask.shape=}, {sam_mask.shape=}, {rgb_img.shape=}, {depth_img.shape=}")
        masked_depth = depth_img[None, ...] * sam_mask
        masked_rgb = rgb_img[None, ...] * sam_mask[..., None]
        return masked_depth, masked_rgb

    def get_pcd(self, pts, cls, transformation_matrix, debug):
        # build Open3D PointCloud
        if debug:
            print(f"\n\n[SAM2_PC get_pcd_bbox] {len(pts)=} {len(cls)=}")
        pcd = o3tg.PointCloud(self.o3dDevice)
        pcd.point["positions"] = o3c.Tensor(pts.numpy().astype(np.float32), o3c.float32, self.o3dDevice)
        pcd.point["colors"] = o3c.Tensor((cls.numpy()/255.0).astype(np.float32), o3c.float32, self.o3dDevice)
        pcd = pcd.voxel_down_sample(voxel_size=config["voxel_size"])
        # Apply statistical outlier removal to denoise the point cloud
        if config["statistical_outlier_removal"]:
            pcd, ind = pcd.remove_statistical_outliers(nb_neighbors=config["statistical_nb_neighbors"], std_ratio=config["statistical_std_ratio"])
        if config["radius_outlier_removal"]:
            pcd, ind = pcd.remove_radius_outliers(nb_points=config["radius_nb_points"], search_radius=config["radius_radius"])
        pcd = pcd.transform(transformation_matrix)
        
        return pcd


    def predict(self, rgb_img, depth_img, bbox, probs, intrinsics, obs_pose, debug, query_str=""):
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
        - pcds: List of reduced point clouds
        - bounding_boxes_3d: List of reduced 3D bounding boxes
        - new_probs: List of reduced scores
        - masked_rgb: List of RGB masks
        - masked_depth: List of Depth Masks
        """
        if debug:
            print(f"[SAM2_PC predict] Received {len(bbox)=}, {len(probs)=}")
            print(f"[SAM2_PC predict] query_str = {query_str}")
        masked_depth, masked_rgb = self.get_masks(rgb_img, depth_img, bbox, debug=debug)
        if masked_depth is None or masked_rgb is None:
            return [], [], torch.tensor([]), [], []
        if debug:
            print(f"[SAM2_PC predict] masked_depth.shape = {masked_depth.shape}")
            print(f"[SAM2_PC predict] masked_rgb.shape   = {masked_rgb.shape}")
        tensor_depth = torch.from_numpy(masked_depth).to(self.device)
        tensor_rgb = torch.from_numpy(masked_rgb).to(self.device)

        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        points, colors = get_points_and_colors(tensor_depth, tensor_rgb, fx, fy, cx, cy)
        colors = colors[..., [2, 1, 0]]
        if debug:
            print(f"[SAM2_PC predict] {points.shape=}, {colors.shape=}")

        B, N, _ = points.shape

        pcds = []
        bounding_boxes_3d = []
        new_masked_rgb = []
        new_masked_depth = []
        new_probs = []

        pts_cpu = points.detach().cpu()
        cols_cpu = colors.detach().cpu()
        #for each candiate object get the point cloud unless there are too few points
        transformation_matrix = pose_to_matrix(obs_pose)
        for i in range(B):
            pts = pts_cpu[i]
            cls = cols_cpu[i]

            # mask out void points
            depths = pts[:, 2]
            valid = (depths > config["camera_min_range_m"]) & (depths < config["camera_max_range_m"])
            if debug:
                print(f"[SAM2_PC predict] {valid.sum()=}")
            if valid.sum() < config["min_3d_points"]:
                continue
            pts = pts[valid]
            cls = cls[valid]
            pcd = self.get_pcd(pts, cls, transformation_matrix, debug=debug)
            if pcd.point["positions"].shape[0] < config["min_3d_points"]:
                continue
            bbox = pcd.get_axis_aligned_bounding_box()
            if debug:
                print(f"[SAM2_PC get_pcd_bbox] {pcd=}")
                print(f"[SAM2_PC get_pcd_bbox] {bbox=}")

            pcds.append(pcd)
            bounding_boxes_3d.append(bbox)
            new_masked_rgb.append(masked_rgb[i])
            new_masked_depth.append(masked_depth[i])
            new_probs.append(probs[i])

        return pcds, bounding_boxes_3d, new_probs, masked_rgb, masked_depth
    def __str__(self):
        return f"SAM2: {self.sam_predictor.model.device}"
    def __repr__(self):
        return self.__str__()

if __name__ == "__main__":
    from BBBackBones import OWLv2, YOLO_WORLD, Gemini_BB
    from realsense_devices import vis_loop


    selected_model = input("input yolo for yolo world, gemini for gemini, owl for Owlv2: ")
    if selected_model == "gemini":
        bb = Gemini_BB()
    elif selected_model == "owl":
        bb = OWLv2()
    elif selected_model == "yolo":
        bb = YOLO_WORLD()
    else:
        raise ValueError(f"Unknown model {selected_model}")

    sam2 = SAM2_PC()
    

    queries = config["test_querys"]


    do_debug = False
    # ---- RealSense grabber thread (no GUI calls here) ----
    def grabber_func(cam):
        data = cam.get_data()
        fx=data['color_intrinsics'].fx
        fy=data['color_intrinsics'].fy
        cx=data['color_intrinsics'].ppx
        cy=data['color_intrinsics'].ppy
        intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        data = cam.get_data()  # make sure this has a short internal timeout
        rgb_img = data["color_img"]
        depth_img = data["depth_img"] * cam.depth_scale
        bb_results = bb.predict(data["color_img"], queries, debug=do_debug)
        tmp_latest_pcs = {}
        tmp_latest_bbs = {}
        tmp_latest_probs = {}
        for q in queries:
            if q not in bb_results:
                continue
            pcds, bboxes, probs, masked_rgb, masked_depth = sam2.predict(rgb_img, depth_img, bb_results[q]["boxes"], bb_results[q]["probs"], intrinsics, [0,0,0,0,0,0], debug=do_debug, query_str=q)
            if len(pcds) == 0:
                # nothing valid this tick for this query
                continue
            # print(f"{probs.shape=}")
            tmp_latest_pcs[q] = pcds
            tmp_latest_bbs[q] = bboxes
            tmp_latest_probs[q] = probs
        
        out_dict = {
            "pcs": tmp_latest_pcs,
            "bbs": tmp_latest_bbs,
            "probs": tmp_latest_probs,
            "vertices": data["vertices"],
            "colors": data["colors"]
        }
        return out_dict



    vis_loop(grabber_func)
        
