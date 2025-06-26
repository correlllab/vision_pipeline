import open3d as o3d
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import time

import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)

from utils import iou_3d, pose_to_matrix, matrix_to_pose, in_image
from FoundationModels import OWLv2, SAM2_PC, display_owl, display_sam2
_script_dir = os.path.dirname(os.path.realpath(__file__))
_config_path = os.path.join(_script_dir, 'config.json')
fig_dir = os.path.join(_script_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(os.path.join(fig_dir, "VP"), exist_ok=True)
config = json.load(open(_config_path, 'r'))

class VisionPipe:
    def __init__(self):
        """
        Initializes the VisionPipe with OWLv2 and SAM2_PC models.
        Sets up the tracked objects dictionary to store 3D predictions.
        tracked_objects[object_name] = {
            "boxes": List of 3D bounding boxes,
            "scores": Tensor of belief scores,
            "pcds": List of point clouds,
            "rgb_masks": List of lists of RGB masks,
            "depth_masks": List of lists of depth masks
            "names": List of strings for object names
        }
        """
        self.owv2 = OWLv2()
        self.sam2 = SAM2_PC()
        self.tracked_objects = {}
        self.pose_time_tracker = ()
        self.update_count = 0

    def update(self, rgb_img, depth_img, querries, I, obs_pose, time_stamp, debug = False):
        """
        Generates a set of 3D predictions and then updates the tracked objects based on the new observations.
        candidates_3d[object_name] = {
            "boxes": List of 3D bounding boxes,
            "scores": Tensor of belief scores,
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
        

        #get 2d predictions dict with lists of scores, boxes from OWLv2
        candidates_2d = self.owv2.predict(rgb_img, querries, debug=debug)
        if debug:
            display_owl(rgb_img, candidates_2d)

        #prepare the 3d predictions dict
        candidates_3d = {}
        #Will need to transform points according to robot pose
        transformation_matrix = pose_to_matrix(obs_pose)
        for object, candidate in candidates_2d.items():
            #convert each set of [boxes, scores] to 3D point clouds
            pcds, box_3d, scores, rgb_masks, depth_masks = self.sam2.predict(rgb_img, depth_img, candidate["boxes"], candidate["scores"], I, debug=debug)
            pcds = [pcd.transform(transformation_matrix) for pcd in pcds]
            box_3d = [pcd.get_axis_aligned_bounding_box() for pcd in pcds]
            #print(f"{box_3d=}")
            if debug:
                display_sam2(pcds, box_3d, scores, window_prefix=f"{object} ")

            #populate the candidates_3d dict
            candidates_3d[object] = {"boxes": box_3d, "scores": scores, "pcds": pcds, "rgb_masks": rgb_masks, "depth_masks": depth_masks}
            if debug:
                print(f"{object=}")
                print(f"   {len(candidates_3d[object]['boxes'])=}, {len(candidates_3d[object]['pcds'])=}, {candidates_3d[object]['scores'].shape=}, {candidates_3d[object]['rgb_masks'].shape=}, {candidates_3d[object]['depth_masks'].shape=}")
        #update the tracked objects with the new predictions
        self.update_tracked_objects(candidates_3d, obs_pose, I, debug=debug)
        self.remove_low_belief_objects()
        self.update_count += 1
        self.pose_time_tracker.append((obs_pose, time_stamp))
        return True, "successfully updated"

    def update_tracked_objects(self, candidates_3d, obs_pose, I, debug = False):
        """
        Given a set of 3D candidates, updates the tracked objects dictionary.
        tracked_objects[object_name] = {
            "boxes": List of 3D bounding boxes,
            "scores": Tensor of belief scores,
            "pcds": List of point clouds,
            "rgb_masks": List of lists of RGB masks,
            "depth_masks": List of lists of depth masks
            "names": List of strings for object names
        }
        candidates_3d[object_name] = {
            "boxes": List of 3D bounding boxes,
            "scores": Tensor of belief scores,
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
            for candidate_box, candidate_score, candidate_pcd, candidate_rgb_mask, candidate_depth_mask in zip(candidates["boxes"], candidates["scores"], candidates["pcds"], candidates["rgb_masks"], candidates["depth_masks"]):
                #calculate the 3D IoU with all tracked objects
                ious = [iou_3d(candidate_box, box) for box in self.tracked_objects[object]["boxes"][:n_preupdated_tracked_objects]]
                max_iou = max(ious)
                #If the max IoU is below the threshold, add the candidate as a new object
                if max_iou < config["iou_3d_matching"]:
                    self.tracked_objects[object]["boxes"].append(candidate_box)
                    self.tracked_objects[object]["pcds"].append(candidate_pcd)
                    self.tracked_objects[object]["rgb_masks"].append([candidate_rgb_mask])
                    self.tracked_objects[object]["depth_masks"].append([candidate_depth_mask])

                    # your existing scores (shape: torch.Size([32]))
                    scores = self.tracked_objects[object]['scores']

                    # make it a 1-element tensor
                    new_score = candidate_score.unsqueeze(0)   # shape: [1]

                    # concatenate along dim=0
                    updated_scores = torch.cat([scores, new_score], dim=0)  # shape: [33]

                    # store it back
                    self.tracked_objects[object]['scores'] = updated_scores
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
                    b = self.tracked_objects[object]["scores"][match_idx]
                    x = candidate_score

                    # weighted‐power update
                    num = b * x
                    den = num + ((1-b) * (1-x))

                    self.tracked_objects[object]["scores"][match_idx] = num / den
                    updated[match_idx] = True
            # If the object was not updated but it should've been, we need to update its belief score
            for i, (tracked_score, pcd, obj_updated) in enumerate(zip(self.tracked_objects[object]["scores"], self.tracked_objects[object]["pcds"], updated)):
                if obj_updated:
                    continue
                centroid = pcd.get_center()

                if not in_image(centroid, obs_pose, I): #or obscured:
                    continue
                p_fn = config["false_negative_rate"]

                num = tracked_score * p_fn
                den = num + ((1-tracked_score) * (1-p_fn))
                new_score = num/den

                self.tracked_objects[object]["scores"][i] = new_score

    def remove_low_belief_objects(self):
        """
        Removes objects from the tracked_objects dictionary that have a belief score below the threshold.
        """
        for object, prediction in self.tracked_objects.items():

            mask = prediction["scores"] > config["belief_threshold"]
            prediction["scores"] = prediction["scores"][mask]
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
            argmax1, maxval1 = max(enumerate(candiates["scores"]), key=lambda pair: pair[1])
            top_candiate = candiates["pcds"][argmax1]
            return top_candiate, maxval1
        raise ValueError(f"Object {query} not found in tracked objects.")

    def vis_belief2D(self, query, blocking=True, n_rows = 10, n_cols = 5, prefix="", save_dir=None):
        """
        Visualizes the belief scores of a given object in a bar plot and its RGB masks in a grid.
        """
        if query not in self.tracked_objects:
            raise ValueError(f"Object {query} not found in tracked objects.")
        bundles = [(self.tracked_objects[query]["scores"][i], self.tracked_objects[query]["rgb_masks"][i]) for i in range(len(self.tracked_objects[query]["scores"]))]

        scores = [bundle[0] for bundle in bundles]
        # Create a bar plot of the scores
        indices = np.arange(len(scores))
        plt.figure(figsize=(8, 4))
        plt.bar(indices, scores.cpu().numpy() if hasattr(scores, "cpu") else scores, color='skyblue')
        plt.xlabel("Belief Rank")
        plt.ylabel("Belief Score")
        plt.title(f"{prefix}Belief Scores for '{query}'")
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"{prefix}_{query}_belief_scores.png"))

        # Plot n_cols RGB masks in a grid of the top n_rows candidates
        n_rows = min(n_rows, len(bundles))
        max_imgs = max([len(rgb_masks) for _, rgb_masks in bundles[:n_rows]])
        n_cols = min(n_cols, max_imgs)
        n_cols = max(n_cols, 2)  # Ensure at least one column
        #print(f"Visualizing {query} with {n_rows} rows and {n_cols} columns of images.")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(19,12))
        fig.suptitle(f"{prefix}Belief Scores for '{query}'", fontsize=16)
        for i, (score, rgb_masks)in enumerate(bundles[:n_rows]):
            axes[i, 0].set_title(f"{query} {i} Score: {score:.2f}")
            for j in range(min(len(rgb_masks), n_cols)):
                axes[i, j].imshow(rgb_masks[j])
                axes[i, j].axis('off')
        fig.tight_layout()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"{prefix}_{query}_belief_views.png"))
        plt.show(block=blocking)
        #plt.pause(0.1)


def test_VP(display2d=True):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    profile = pipeline.start(cfg)

    sensor = profile.get_device().first_depth_sensor()
    depth_scale = sensor.get_depth_scale()

    align = rs.align(rs.stream.color)

    video_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = video_prof.get_intrinsics()



    vp = VisionPipe()
    for i in range(10):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        rgb_img = np.asanyarray(color_frame.get_data())


        depth_frame = aligned.get_depth_frame()
        depth_img = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_img *= depth_scale


        I = {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
            "width": rgb_img.shape[1],
            "height": rgb_img.shape[0],
        }

        print(f"Frame {i}:")
        #print(f"   {rgb_img.shape=}, {depth_img.shape=}, {I=}")
        predictions = vp.update(rgb_img, depth_img, config["test_querys"], I, [0.0]*6)
        if display2d:
            vp.vis_belief2D(query=config["test_querys"][0], blocking=True, prefix=f"T={i}", save_dir=os.path.join(fig_dir, "VP"))

        for object, prediction in predictions.items():
            print(f"{object=}")
            print(f"   {len(prediction['boxes'])=}, {len(prediction['pcds'])=}, {prediction['scores'].shape=}")
        print(f"\n\n")
    if display2d:
        vp.vis_belief2D(query=config["test_querys"][0], blocking=True, prefix= "Final",save_dir=os.path.join(fig_dir, "VP"))



if __name__ == "__main__":
    print(f"\n\nTESTING VP")
    test_VP()
