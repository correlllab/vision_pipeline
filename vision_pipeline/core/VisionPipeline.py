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


from math_utils import iou_3d, in_image, is_obscured
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
        elif config["backbone"] == "yoloworld":
            self.BackBone = YOLO_WORLD()
        else:
            raise ValueError(f"Unknown backbone {config['backbone']=}. Please choose 'gemini','owl' or 'yoloworld'.")
        self.sam2 = SAM2_PC()
        self.tracked_objects = {}
        self.pose_time_tracker = ()
        self.update_count = 0

    def update(self, rgb_img, depth_img, queries, I, obs_pose, time_stamp, debug):
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
        queries = [q.lower() for q in queries]
        #throw away old poses
        self.pose_time_tracker = [(pose, time) for pose, time in self.pose_time_tracker if time > (time_stamp - config["pose_expire_time"])]

        #if the current call to update is too close to a prior, non expired call to update
        pose_distances = [np.linalg.norm(np.array(obs_pose) - np.array(pose)) for pose, _ in self.pose_time_tracker]
        if len(pose_distances) > 0 and min(pose_distances) < config["change_in_pose_threshold"] and self.update_count > 0:
            return False, "Pose too close in time and distance to a previous update"

        candidates_3d = self.get_candidates(rgb_img, depth_img, queries, I, obs_pose, debug)
        
        #update the tracked objects with the new predictions
        self.update_tracked_objects(candidates_3d, obs_pose, I, rgb_img, depth_img, debug=debug)
        self.remove_low_belief_objects()
        self.update_count += 1
        self.pose_time_tracker.append((obs_pose, time_stamp))
        return True, "successfully updated"
    
    def get_candidates(self, rgb_img, depth_img, queries, I, obs_pose, debug):
        #get 2d predictions dict with lists of probabilities, boxes from OWLv2
        candidates_2d = self.BackBone.predict(rgb_img, queries, debug=debug)

        #prepare the 3d predictions dict
        candidates_3d = {}
        #Will need to transform points according to robot pose
        for object_str, candidate in candidates_2d.items():
            #convert each set of [boxes, probs] to 3D point clouds
            pcds, box_3d, probs, rgb_masks, depth_masks = self.sam2.predict(rgb_img, depth_img, candidate["boxes"], candidate["probs"], I, obs_pose, debug=debug, query_str=object_str)

            #populate the candidates_3d dict
            candidates_3d[object_str] = {"boxes": box_3d, "probs": probs, "pcds": pcds, "rgb_masks": rgb_masks, "depth_masks": depth_masks}
            if debug:
                print(f"[VisionPipe update]{object_str=}")
                print(f"   {len(candidates_3d[object_str]['boxes'])=}, {len(candidates_3d[object_str]['pcds'])=}, {len(candidates_3d[object_str]['probs'])=}, {len(candidates_3d[object_str]['rgb_masks'])=}, {len(candidates_3d[object_str]['depth_masks'])=}")
        return candidates_3d

    def belief_update(self, belief_prob, sample_prob, sample_weight = 1):
        # Apply the weight to the sample probability
        # A weight < 1 pulls the probability towards 0.5, reducing its impact
        weighted_sample_prob = sample_prob ** sample_weight

        # Proceed with the Bayesian update using the weighted probability
        num = belief_prob * weighted_sample_prob
        den = num + ((1 - belief_prob) * (1 - weighted_sample_prob))

        # Avoid division by zero if den is 0
        # if den == 0:
        #     return 0.0
        #should never be 0
        
        new_prob = num / den
        return new_prob

    def merge_candidate(self, candidate_box, candidate_prob, candidate_pcd, candidate_rgb_mask, candidate_depth_mask, obj_str, n_considered):
        #calculate the 3D IoU with all tracked objects
        #USE IQR to determine matches
        ious = [iou_3d(candidate_box, box) for box in self.tracked_objects[obj_str]["boxes"][:n_considered]]
        try:
            max_iou = max(ious)
        except Exception as e:
            print(f"[VisionPipe update_tracked_objects] Exception {e} with ious {ious=} for object {obj_str} {len(self.tracked_objects[obj_str]['boxes'])=} {len(self.tracked_objects[obj_str]['pcds'])=} {n_considered=}")
            raise ValueError(f"[VisionPipe update_tracked_objects] Exception {e} with ious {ious=} for object {obj_str} {len(self.tracked_objects[obj_str]['boxes'])=} {len(self.tracked_objects[obj_str]['pcds'])=} {n_considered=}")
        q1 = np.percentile(ious, 25)
        q3 = np.percentile(ious, 75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        match_idx = None
        #If the max IoU is below the threshold, add the candidate as a new object
        if max_iou < upper_bound:
            self.tracked_objects[obj_str]["boxes"].append(candidate_box)
            self.tracked_objects[obj_str]["pcds"].append(candidate_pcd)
            self.tracked_objects[obj_str]["rgb_masks"].append([candidate_rgb_mask])
            self.tracked_objects[obj_str]["depth_masks"].append([candidate_depth_mask])

            # your existing scores (shape: torch.Size([32]))
            probs = self.tracked_objects[obj_str]['probs']

            # make it a 1-element tensor
            new_prob = candidate_prob.unsqueeze(0)   # shape: [1]

            # concatenate along dim=0
            updated_probs = torch.cat([probs, new_prob], dim=0)  # shape: [33]

            # store it back
            self.tracked_objects[obj_str]['probs'] = updated_probs
            previous_names = self.tracked_objects[obj_str]["names"][-1]
            next_idx = int(previous_names.split("_")[-1]) + 1
            self.tracked_objects[obj_str]["names"].append(f"{obj_str}_{next_idx}")

        # If the max IoU is above the threshold, update the existing object
        else:
            match_idx = ious.index(max_iou)
            pcd = self.tracked_objects[obj_str]["pcds"][match_idx] + candidate_pcd
            pcd = pcd.voxel_down_sample(voxel_size=config["voxel_size"])
            # Apply statistical outlier removal to denoise the point cloud
            if config["statistical_outlier_removal"]:
                pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=config["statistical_nb_neighbors"], std_ratio=config["statistical_std_ratio"])
            if config["radius_outlier_removal"]:
                pcd, ind = pcd.remove_radius_outlier(nb_points=config["radius_nb_points"], radius=config["radius_radius"])
            self.tracked_objects[obj_str]["pcds"][match_idx] = pcd
            self.tracked_objects[obj_str]["boxes"][match_idx] = pcd.get_axis_aligned_bounding_box()
            self.tracked_objects[obj_str]["rgb_masks"][match_idx].append(candidate_rgb_mask)
            self.tracked_objects[obj_str]["depth_masks"][match_idx].append(candidate_depth_mask)
            #print(f"Updating {object} with match_idx {match_idx} {len(self.tracked_objects[object]['rgb_masks'])=}")
            # existing belief and new observation

            self.tracked_objects[obj_str]["probs"][match_idx] = self.belief_update(self.tracked_objects[obj_str]["probs"][match_idx], candidate_prob)
        return match_idx

    def update_tracked_objects(self, candidates_3d, obs_pose, I, rgb_img, depth_img, debug):
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
        for obj_str, candidates in candidates_3d.items():
            #if it wasnt being tracked before add all candidates to tracked objects
            if obj_str not in self.tracked_objects:
                self.tracked_objects[obj_str] = candidates
                self.tracked_objects[obj_str]["rgb_masks"] = [[rgb_mask] for rgb_mask in candidates["rgb_masks"]]
                self.tracked_objects[obj_str]["depth_masks"] = [[depth_mask] for depth_mask in candidates["depth_masks"]]
                self.tracked_objects[obj_str]["names"] = [f"{obj_str}_{i}" for i in range(len(candidates["boxes"]))]
                continue #object was not tracked before, so we just add it

            #keep track of which objects were updated, if an object was not updated but should have been, we will update its belief score
            n_preupdated_tracked_objects = len(self.tracked_objects[obj_str]["boxes"])
            updated = [False] * n_preupdated_tracked_objects
            #for each candidate
            for candidate_box, candidate_prob, candidate_pcd, candidate_rgb_mask, candidate_depth_mask in zip(candidates["boxes"], candidates["probs"], candidates["pcds"], candidates["rgb_masks"], candidates["depth_masks"]):
                merge_idx = self.merge_candidate(candidate_box, candidate_prob, candidate_pcd, candidate_rgb_mask, candidate_depth_mask, obj_str, n_preupdated_tracked_objects)
                if merge_idx is not None:
                    updated[merge_idx] = True

            # If the object was not updated but it should've been, we need to update its belief score
            for i, (tracked_prob, pcd, obj_updated) in enumerate(zip(self.tracked_objects[obj_str]["probs"], self.tracked_objects[obj_str]["pcds"], updated)):
                if obj_updated:
                    continue
                centroid = pcd.get_center()

                if not in_image(centroid, obs_pose, I):
                    continue
                if is_obscured(pcd, depth_img, obs_pose, I):
                    continue

                p_fn = config["vlm_false_negative_rate"]

                self.tracked_objects[obj_str]["probs"][i] = self.belief_update(tracked_prob, p_fn)

    def remove_low_belief_objects(self):
        """
        Removes objects from the tracked_objects dictionary that have a belief score below the threshold.
        """
        for obj_str, prediction in self.tracked_objects.items():

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

            self.tracked_objects[obj_str] = prediction
    def query(self, query):
        """
        Returns the top candidate point cloud and its belief score for a given object.
        """
        if query in self.tracked_objects and len(self.tracked_objects[query]["pcds"]) > 0 and len(self.tracked_objects[query]["probs"]) > 0:
            candiates = self.tracked_objects[query]
            #print(f"{candiates=}")
            argmax1, maxval1 = max(enumerate(candiates["probs"]), key=lambda pair: pair[1])
            top_candiate = candiates["pcds"][argmax1]
            return top_candiate, maxval1
        return o3d.geometry.PointCloud(), 0.0




if __name__ == "__main__":
    from realsense_devices import RealSenseCamera
    import open3d.visualization.gui as gui
    import threading

    vp = VisionPipe()
    camera = RealSenseCamera()
    data = camera.get_data()

    fx=data['color_intrinsics'].fx
    fy=data['color_intrinsics'].fy
    cx=data['color_intrinsics'].ppx
    cy=data['color_intrinsics'].ppy
    intrinsics = {"fx": fx, "fy": fx, "cx": cx, "cy": cy, "width": data['color_intrinsics'].width, "height": data['color_intrinsics'].height,}
    obs_pose = [0, 0, 0, 0, 0, 0]  # Example pose
    queries = config["test_querys"]
    queries.append("ghost")

    try:
        # Initialize the GUI application
        app = gui.Application.instance
        app.initialize()

        # Create the visualizer window
        vis = o3d.visualization.O3DVisualizer("RealSense 3D Viewer", 1024, 768)
        vis.show_settings = True

        # Add a coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry("frame", coord_frame)

        
        # Set a default camera view
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run_one_tick()

        last_update = vp.update_count
        def thread_func():
            last_update_time = time.time() - config["pose_expire_time"]-1
            while vis.is_visible and camera.open:
                data = camera.get_data()
                rgb_img = data["color_img"]
                depth_img = data["depth_img"]*camera.depth_scale
                if time.time() - last_update_time > config["pose_expire_time"]:    
                    print("\n\n")
                    success, message = vp.update(rgb_img, depth_img, queries, intrinsics, obs_pose, time_stamp=time.time(), debug=False)
                    print(f"Update success: {success}, message: {message}")
                    if success:
                        last_update_time = time.time()
                    print("\n\n")
        thread = threading.Thread(target=thread_func, daemon=True)
        thread.start()
        print("\n\n\n")
        #while vis.is_visible:
        i = 0
        while i < 10:
            if vp.update_count != last_update:
                last_update = vp.update_count
                vis.clear_3d_labels()

                for q in queries:
                    q_pcd, q_prob = vp.query(q)
                    if q_prob == 0.0:
                        continue
                    bbox = q_pcd.get_axis_aligned_bounding_box()
                    q_pcd = q_pcd.voxel_down_sample(voxel_size=0.001)
                    vis.remove_geometry(f"{q}_pcd")
                    vis.remove_geometry(f"{q}_bb")
                    vis.add_geometry(f"{q}_pcd", q_pcd)
                    vis.add_geometry(f"{q}_bb", bbox)
                    vis.add_3d_label(q_pcd.get_center(), f"{q} {q_prob:.2f}")
                i+= 1
            app.run_one_tick()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if camera:
            camera.stop()
        vis.close()
        app.quit()
    

    
    
        