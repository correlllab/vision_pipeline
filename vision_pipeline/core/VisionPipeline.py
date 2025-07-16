import open3d as o3d
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import time

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
os.makedirs(os.path.join(fig_dir, "VisionPipeline"), exist_ok=True)


from math_utils import iou_3d, pose_to_matrix, matrix_to_pose, in_image, is_obscured
from SAM2 import SAM2_PC
from BBBackBones import OWLv2, Gemini_BB, YOLO_WORLD

class VisionPipe:
    def __init__(self):
        """
        Initializes the VisionPipe with OWLv2 and SAM2_PC models.
        Sets up the tracked objects dictionary to store 3D predictions.
        tracked_objects[object_name] = {
            "boxes": List of 3D bounding boxes,
            "probs": Tensor of belief probs,
            "pcds": List of point clouds,
            "rgb_masks": List of lists of RGB masks,
            "depth_masks": List of lists of depth masks
            "names": List of strings for object names
        }
        """
        if config["backbone"] == "gemini":
            self.BackBone = Gemini_BB()
        elif config["backbone"] == "owl":
            self.BackBone = OWLv2()
        elif config["backbone"] == "YOLOWORLD":
            self.BackBone = YOLO_WORLD()
        else:
            raise ValueError(f"Unknown backbone {config['backbone']=}. Please choose 'gemini' or 'owl'.")
        self.sam2 = SAM2_PC()
        self.tracked_objects = {}
        self.pose_time_tracker = ()
        self.update_count = 0

    def update(self, rgb_img, depth_img, querries, I, obs_pose, time_stamp, debug = False):
        """
        Generates a set of 3D predictions and then updates the tracked objects based on the new observations.
        candidates_3d[object_name] = {
            "boxes": List of 3D bounding boxes,
            "probs": Tensor of belief probs,
            "pcds": List of point clouds,
            "rgb_masks": List of RGB masks,
            "depth_masks": List of depth masks
        }
        """
        #throw away old poses
        self.pose_time_tracker = [(pose, time) for pose, time in self.pose_time_tracker if time > (time_stamp - config["pose_expire_time"])]

        #if the current call to update is too close to a prior, non expired call to update
        pose_distances = [np.linalg.norm(np.array(obs_pose) - np.array(pose)) for pose, _ in self.pose_time_tracker]
        #print(f"Pose distances: {pose_distances}")
        if len(pose_distances) > 0 and min(pose_distances) < config["change_in_pose_threshold"] and self.update_count > 0:
            return False, "too close to previous update"


        #get 2d predictions dict with lists of probabilities, boxes from OWLv2
        candidates_2d = self.BackBone.predict(rgb_img, querries, debug=debug)


        #prepare the 3d predictions dict
        candidates_3d = {}
        #Will need to transform points according to robot pose
        transformation_matrix = pose_to_matrix(obs_pose)
        for object, candidate in candidates_2d.items():
            #convert each set of [boxes, probs] to 3D point clouds
            pcds, box_3d, probs, rgb_masks, depth_masks = self.sam2.predict(rgb_img, depth_img, candidate["boxes"], candidate["probs"], I, debug=debug, query_str=object)
            pcds = [pcd.transform(transformation_matrix) for pcd in pcds]
            box_3d = [pcd.get_axis_aligned_bounding_box() for pcd in pcds]
            #print(f"{box_3d=}")


            #populate the candidates_3d dict
            candidates_3d[object] = {"boxes": box_3d, "probs": probs, "pcds": pcds, "rgb_masks": rgb_masks, "depth_masks": depth_masks}
            if debug:
                print(f"{object=}")
                print(f"   {len(candidates_3d[object]['boxes'])=}, {len(candidates_3d[object]['pcds'])=}, {len(candidates_3d[object]['probs'])=}, {len(candidates_3d[object]['rgb_masks'])=}, {len(candidates_3d[object]['depth_masks'])=}")
        #update the tracked objects with the new predictions
        self.update_tracked_objects(candidates_3d, obs_pose, I, rgb_img, depth_img, debug=debug)
        self.remove_low_belief_objects()
        self.update_count += 1
        self.pose_time_tracker.append((obs_pose, time_stamp))
        return True, "successfully updated"

    def update_tracked_objects(self, candidates_3d, obs_pose, I, rgb_img, depth_img, debug = False):
        """
        Given a set of 3D candidates, updates the tracked objects dictionary.
        tracked_objects[object_name] = {
            "boxes": List of 3D bounding boxes,
            "probs": Tensor of belief probs,
            "pcds": List of point clouds,
            "rgb_masks": List of lists of RGB masks,
            "depth_masks": List of lists of depth masks
            "names": List of strings for object names
        }
        candidates_3d[object_name] = {
            "boxes": List of 3D bounding boxes,
            "probs": Tensor of belief probs,
            "pcds": List of point clouds,
            "rgb_masks": List of RGB masks,
            "depth_masks": List of depth masks
        }
        """
        #for each candidate object in candidates_3d
        for object, candidates in candidates_3d.items():
            #if it wasnt being tracked before add all candidates to tracked objects
            if object not in self.tracked_objects:
                self.tracked_objects[object] = candidates
                self.tracked_objects[object]["rgb_masks"] = [[rgb_mask] for rgb_mask in candidates["rgb_masks"]]
                self.tracked_objects[object]["depth_masks"] = [[depth_mask] for depth_mask in candidates["depth_masks"]]
                self.tracked_objects[object]["names"] = [f"{object}_{i}" for i in range(len(candidates["boxes"]))]
                continue #object was not tracked before, so we just add it

            #keep track of which objects were updated, if an object was not updated but should have been, we will update its belief score
            n_preupdated_tracked_objects = len(self.tracked_objects[object]["boxes"])
            updated = [False] * n_preupdated_tracked_objects
            #for each candidate
            for candidate_box, candidate_prob, candidate_pcd, candidate_rgb_mask, candidate_depth_mask in zip(candidates["boxes"], candidates["probs"], candidates["pcds"], candidates["rgb_masks"], candidates["depth_masks"]):
                #calculate the 3D IoU with all tracked objects
                #USE IQR to determine matches
                ious = [iou_3d(candidate_box, box) for box in self.tracked_objects[object]["boxes"][:n_preupdated_tracked_objects]]
                max_iou = max(ious)
                q1 = np.percentile(ious, 25)
                q3 = np.percentile(ious, 75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                #If the max IoU is below the threshold, add the candidate as a new object
                if max_iou < upper_bound:
                    self.tracked_objects[object]["boxes"].append(candidate_box)
                    self.tracked_objects[object]["pcds"].append(candidate_pcd)
                    self.tracked_objects[object]["rgb_masks"].append([candidate_rgb_mask])
                    self.tracked_objects[object]["depth_masks"].append([candidate_depth_mask])

                    # your existing scores (shape: torch.Size([32]))
                    probs = self.tracked_objects[object]['probs']

                    # make it a 1-element tensor
                    new_prob = candidate_prob.unsqueeze(0)   # shape: [1]

                    # concatenate along dim=0
                    updated_probs = torch.cat([probs, new_prob], dim=0)  # shape: [33]

                    # store it back
                    self.tracked_objects[object]['probs'] = updated_probs
                    previous_names = self.tracked_objects[object]["names"][-1]
                    next_idx = int(previous_names.split("_")[-1]) + 1
                    self.tracked_objects[object]["names"].append(f"{object}_{next_idx}")


                # If the max IoU is above the threshold, update the existing object
                else:
                    match_idx = ious.index(max_iou)
                    pcd = self.tracked_objects[object]["pcds"][match_idx] + candidate_pcd
                    pcd = pcd.voxel_down_sample(voxel_size=config["voxel_size"])
                    # Apply statistical outlier removal to denoise the point cloud
                    if config["statistical_outlier_removal"]:
                        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=config["statistical_nb_neighbors"], std_ratio=config["statistical_std_ratio"])
                    if config["radius_outlier_removal"]:
                        pcd, ind = pcd.remove_radius_outlier(nb_points=config["radius_nb_points"], radius=config["radius_radius"])
                    self.tracked_objects[object]["boxes"][match_idx] = self.tracked_objects[object]["pcds"][match_idx].get_axis_aligned_bounding_box()
                    self.tracked_objects[object]["rgb_masks"][match_idx].append(candidate_rgb_mask)
                    self.tracked_objects[object]["depth_masks"][match_idx].append(candidate_depth_mask)
                    #print(f"Updating {object} with match_idx {match_idx} {len(self.tracked_objects[object]['rgb_masks'])=}")
                    # existing belief and new observation
                    b = self.tracked_objects[object]["probs"][match_idx]
                    x = candidate_prob

                    # weighted‐power update
                    num = b * x
                    den = num + ((1-b) * (1-x))

                    self.tracked_objects[object]["probs"][match_idx] = num / den
                    updated[match_idx] = True
            # If the object was not updated but it should've been, we need to update its belief score
            for i, (tracked_prob, pcd, obj_updated) in enumerate(zip(self.tracked_objects[object]["probs"], self.tracked_objects[object]["pcds"], updated)):
                if obj_updated:
                    continue
                centroid = pcd.get_center()

                if not in_image(centroid, obs_pose, I):
                    continue
                if is_obscured(pcd, depth_img, obs_pose, I):
                    continue

                p_fn = config["vlm_false_negative_rate"]

                num = tracked_prob * p_fn
                den = num + ((1-tracked_prob) * (1-p_fn))
                new_prob = num/den

                self.tracked_objects[object]["probs"][i] = new_prob

    def remove_low_belief_objects(self):
        """
        Removes objects from the tracked_objects dictionary that have a belief score below the threshold.
        """
        for object, prediction in self.tracked_objects.items():

            mask = prediction["probs"] > config["remove_belief_threshold"]
            prediction["probs"] = prediction["probs"][mask]
            boxes = []
            pcds = []
            rgb_masks = []
            depth_masks = []
            names = []
            for i, b in enumerate(mask):
                if b:
                    boxes.append(prediction["boxes"][i])
                    pcds.append(prediction["pcds"][i])
                    rgb_masks.append(prediction["rgb_masks"][i])
                    depth_masks.append(prediction["depth_masks"][i])
                    names.append(prediction["names"][i])
            prediction["boxes"] = boxes
            prediction["pcds"] = pcds
            prediction["rgb_masks"] = rgb_masks
            prediction["depth_masks"] = depth_masks
            prediction["names"] = names

            self.tracked_objects[object] = prediction
    def query(self, query):
        """
        Returns the top candidate point cloud and its belief score for a given object.
        """
        if query in self.tracked_objects:
            candiates = self.tracked_objects[query]
            #print(f"{candiates=}")
            argmax1, maxval1 = max(enumerate(candiates["probs"]), key=lambda pair: pair[1])
            top_candiate = candiates["pcds"][argmax1]
            return top_candiate, maxval1
        return o3d.geometry.PointCloud(), 0.0




if __name__ == "__main__":
    import cv2
    from SAM2 import display_3dCandidates
    vp = VisionPipe()
    rgb_img = cv2.imread("./ExampleImages/MushroomRGB.jpeg")
    depth_img = cv2.imread("./ExampleImages/MushroomDepth.png")
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    center_x = rgb_img.shape[1] // 2
    center_y = rgb_img.shape[0] // 2
    intrinsics = {"fx": 500, "fy": 500, "cx": center_x, "cy": center_y, 'width': rgb_img.shape[1], "height":rgb_img.shape[0]}
    obs_pose = [0, 0, 0, 0, 0, 0]  # Example pose
    queries = ["mushroom", "bottle", "cat"]
    success, message = vp.update(rgb_img, depth_img, queries, intrinsics, obs_pose, time_stamp=time.time(), debug=True)
    print(f"Update success: {success}, message: {message}")

    candidates_3d = {}
    queries.append("ghost")
    for q in queries:
        pcd, prob = vp.query(q)
        bbox = pcd.get_axis_aligned_bounding_box()
        candidates_3d[q] = {
            "boxes": [bbox],
            "probs": torch.tensor([prob]),
            "pcds": [pcd],
            "rgb_masks": [[]],
            "depth_masks": [[]]
        }
    display_3dCandidates(candidates_3d, window_prefix="VisionPipe_")