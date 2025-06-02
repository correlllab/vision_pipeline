import open3d as o3d
import torch
from open3d.visualization import gui, rendering
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
from RealsenseInterface import RealSenseCameraSubscriber
import threading
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
        self.update_count = 0

        self.app = gui.Application.instance
        self.vis = None
        def gui_thread(self):
            """
            Initializes the Open3D GUI application in a separate thread.
            This is necessary to avoid blocking the main thread with the GUI.
            """
            #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

            self.app.initialize()
            self.vis = o3d.visualization.O3DVisualizer("VisionPipe GUI", 1024, 768)
            self.vis.show_settings = True
            self.app.add_window(self.vis)
            while self.app.run_one_tick():
                time.sleep(0.02)
            self.vis.close()
            self.app.quit()

        self.gui_thread = threading.Thread(target=gui_thread, args=(self,))
        self.gui_thread.start()
        self.last_pose = [0.0]*6  # Initialize last pose to zero
        self.camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        while self.vis is None:
            time.sleep(0.1)
        self.vis.add_geometry("CameraFrame", self.camera_frame)

    def update(self, rgb_img, depth_img, querries, I, obs_pose, debug = False):
        """
        Generates a set of 3D predictions and then updates the tracked objects based on the new observations.
        predictions3d[object_name] = {
            "boxes": List of 3D bounding boxes,
            "scores": Tensor of belief scores,
            "pcds": List of point clouds,
            "rgb_masks": List of RGB masks,
            "depth_masks": List of depth masks
        }
        """
        self.last_pose = obs_pose
        #get 2d predictions dict with lists of scores, boxes from OWLv2
        predictions_2d = self.owv2.predict(rgb_img, querries, debug=debug)
        if debug:
            display_owl(rgb_img, predictions_2d)

        #prepare the 3d predictions dict
        predictions_3d = {}
        #Will need to transform points according to robot pose
        transformation_matrix = pose_to_matrix(obs_pose)
        for object, prediction_2d in predictions_2d.items():
            #convert each set of [boxes, scores] to 3D point clouds
            pcds, box_3d, scores, rgb_masks, depth_masks = self.sam2.predict(rgb_img, depth_img, prediction_2d["boxes"], prediction_2d["scores"], I, debug=debug)
            pcds = [pcd.transform(transformation_matrix) for pcd in pcds]
            if debug:
                display_sam2(pcds, box_3d, scores, window_prefix=f"{object} ")

            #populate the predictions_3d dict
            predictions_3d[object] = {"boxes": box_3d, "scores": scores, "pcds": pcds, "rgb_masks": rgb_masks, "depth_masks": depth_masks}
            if debug:
                print(f"{object=}")
                print(f"   {len(predictions_3d[object]['boxes'])=}, {len(predictions_3d[object]['pcds'])=}, {predictions_3d[object]['scores'].shape=}, {predictions_3d[object]['rgb_masks'].shape=}, {predictions_3d[object]['depth_masks'].shape=}")
        #update the tracked objects with the new predictions
        self.update_tracked_objects(predictions_3d, obs_pose, I, debug=debug)
        self.update_count += 1

        self.update_gui()
        return self.tracked_objects

    def update_tracked_objects(self, predictions_3d, obs_pose, I, debug = False):
        """
        Given a set of 3D predictions, updates the tracked objects dictionary.
        tracked_objects[object_name] = {
            "boxes": List of 3D bounding boxes,
            "scores": Tensor of belief scores,
            "pcds": List of point clouds,
            "rgb_masks": List of lists of RGB masks,
            "depth_masks": List of lists of depth masks
            "names": List of strings for object names
        }
        predictions3d[object_name] = {
            "boxes": List of 3D bounding boxes,
            "scores": Tensor of belief scores,
            "pcds": List of point clouds,
            "rgb_masks": List of RGB masks,
            "depth_masks": List of depth masks
        }
        """
        #for each candidate object in predictions_3d
        for object, predictions in predictions_3d.items():
            #if it wasnt being tracked before add all candidates to tracked objects
            if object not in self.tracked_objects:
                self.tracked_objects[object] = predictions_3d[object]
                self.tracked_objects[object]["rgb_masks"] = [[rgb_mask] for rgb_mask in predictions_3d[object]["rgb_masks"]]
                self.tracked_objects[object]["depth_masks"] = [[depth_mask] for depth_mask in predictions_3d[object]["depth_masks"]]
                self.tracked_objects[object]["names"] = [f"{object}_{i}" for i in range(len(predictions_3d[object]["boxes"]))]

                for i, (box, pcd, name) in enumerate(zip(self.tracked_objects[object]["boxes"], self.tracked_objects[object]["pcds"], self.tracked_objects[object]["names"])):
                    self.vis.add_geometry(f"box_{name}", box)
                    self.vis.add_geometry(f"pcd_{name}", pcd)
                    

            else:
                #keep track of which objects were updated, if an object was not updated but should have been, we will update its belief score
                updated = [False for i in range(len(self.tracked_objects[object]["boxes"]))]
                #for each candidate
                for candidate_box, candidate_score, candidate_pcd, candidate_rgb_mask, candidate_depth_mask in zip(predictions["boxes"], predictions["scores"], predictions["pcds"], predictions["rgb_masks"], predictions["depth_masks"]):
                    #calculate the 3D IoU with all tracked objects
                    ious = [iou_3d(candidate_box, box) for box in self.tracked_objects[object]["boxes"]]
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

                        name = self.tracked_objects[object]["names"][-1]
                        self.vis.add_geometry(f"box_{name}", candidate_box)
                        self.vis.add_geometry(f"pcd_{name}", candidate_pcd)
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

                    if not in_image(centroid, obs_pose, I):
                       continue
                    p_fn = config["false_negative_rate"]

                    num = tracked_score * p_fn
                    den = num + ((1-tracked_score) * (1-p_fn))
                    new_score = num/den

                    self.tracked_objects[object]["scores"][i] = new_score
        self.remove_low_belief_objects()

    def remove_low_belief_objects(self):
        """
        Removes objects from the tracked_objects dictionary that have a belief score below the threshold.
        """
        for object, prediction in self.tracked_objects.items():
            mask = prediction["scores"] > config["belief_threshold"]
            prediction["boxes"] = [prediction["boxes"][i] for i in range(len(mask)) if mask[i]]
            prediction["pcds"] = [prediction["pcds"][i] for i in range(len(mask)) if mask[i]]
            prediction["rgb_masks"] = [prediction["rgb_masks"][i] for i in range(len(mask)) if mask[i]]
            prediction["depth_masks"] = [prediction["depth_masks"][i] for i in range(len(mask)) if mask[i]]
            prediction["scores"] = prediction["scores"][mask]
            [self.vis.remove_geometry(f"box_{prediction['names'][i]}") for i in range(len(mask)) if mask[i]]
            [self.vis.remove_geometry(f"pcd_{prediction['names'][i]}") for i in range(len(mask)) if mask[i]]

            prediction["names"] = [prediction["names"][i] for i in range(len(mask)) if mask[i]]

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
        bundles = sorted(bundles, key=lambda x: x[0], reverse=True)

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

    def update_gui(self):
        self.vis.clear_3d_labels()
        self.camera_frame.transform(pose_to_matrix(self.last_pose))
        for query in self.tracked_objects.keys():
            for idx, (box, score) in enumerate(zip(self.tracked_objects[query]["boxes"], self.tracked_objects[query]["scores"])):
                center = box.get_center() if hasattr(box, "get_center") else np.asarray(box.vertices).mean(axis=0)
                self.vis.add_3d_label(center, f"{query}: {float(score):.3f}")


def test_VP(sub, display2d=False):
    vp = VisionPipe()
    i =0
    while True:
        rgb_img, depth_img, Intrinsics, Extrinsics = None, None, None, None
        while rgb_img is None or depth_img is None or Intrinsics is None or Extrinsics is None:
            try:
                print("Waiting for RGB-D data...")
                rgb_img, depth_img, Intrinsics, Extrinsics = sub.read(display=False)
                #print(f"Received RGB-D data: {type(rgb_img)}, {type(depth_img)}, {type(Intrinsics)}, {type(Extrinsics)}")
            except KeyboardInterrupt:
                print("KeyboardInterrupt received, exiting...")
                exit(0)

        I = {
            "fx": Intrinsics[0, 0],
            "fy": Intrinsics[1, 1],
            "cx": Intrinsics[0, 2],
            "cy": Intrinsics[1, 2],
            "width":rgb_img.shape[1],
            "height":rgb_img.shape[0],
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
        i+=1
    if display2d:
        vp.vis_belief2D(query=config["test_querys"][0], blocking=True, prefix= "Final",save_dir=os.path.join(fig_dir, "VP"))



if __name__ == "__main__":
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(networkInterface= "lo")

    sub = RealSenseCameraSubscriber(
        channel_name="realsense/camera",
        InitChannelFactory=False
    )
    rgb_img, depth_img, Intrinsics, Extrinsics = None, None, None, None
    while rgb_img is None or depth_img is None or Intrinsics is None or Extrinsics is None:
        try:
            print("Waiting for RGB-D data...")
            rgb_img, depth_img, Intrinsics, Extrinsics = sub.read(display=False)
            #print(f"Received RGB-D data: {type(rgb_img)}, {type(depth_img)}, {type(Intrinsics)}, {type(Extrinsics)}")
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, exiting...")
            exit(0)
    I = {
        "fx": Intrinsics[0, 0],
        "fy": Intrinsics[1, 1],
        "cx": Intrinsics[0, 2],
        "cy": Intrinsics[1, 2]
    }

    print(f"\n\nTESTING VP")
    test_VP(sub)
