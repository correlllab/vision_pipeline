import open3d as o3d
import torch
from open3d.visualization import gui, rendering
import numpy as np
import json
import os
import open3d as o3d
import matplotlib.pyplot as plt


import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)

from RealsenseInterface import RealSenseCamera
from utils import iou_3d, pose_to_matrix, matrix_to_pose, in_image
from FoundationModels import OWLv2, SAM2_PC, display_owl, display_sam2



_script_dir = os.path.dirname(os.path.realpath(__file__))
_config_path = os.path.join(_script_dir, 'config.json')
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
        }
        """
        self.owv2 = OWLv2()
        self.sam2 = SAM2_PC()
        self.tracked_objects = {}
    def update(self, rgb_img, depth_img, querries, I, obs_pose=[0,0,0,0,0,0], debug = False):
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
                    # If the max IoU is above the threshold, update the existing object
                    else:
                        match_idx = ious.index(max_iou)
                        pcd = self.tracked_objects[object]["pcds"][match_idx] + candidate_pcd
                        pcd.voxel_down_sample(voxel_size=config["voxel_size"])
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
    def querry(self, querry):
        """
        Returns the top candidate point cloud and its belief score for a given object.
        """
        if querry in self.tracked_objects:
            candiates = self.tracked_objects[querry]
            #print(f"{candiates=}")
            argmax1, maxval1 = max(enumerate(candiates["scores"]), key=lambda pair: pair[1])
            top_candiate = candiates["pcds"][argmax1]
            return top_candiate, maxval1
        raise ValueError(f"Object {querry} not found in tracked objects.")
    
    def vis_belief2D(self, querry, blocking=True, n_rows = 10, n_cols = 5, prefix=""):
        """
        Visualizes the belief scores of a given object in a bar plot and its RGB masks in a grid.
        """
        if querry not in self.tracked_objects:
            raise ValueError(f"Object {querry} not found in tracked objects.")
        bundles = [(self.tracked_objects[querry]["scores"][i], self.tracked_objects[querry]["rgb_masks"][i]) for i in range(len(self.tracked_objects[querry]["scores"]))]
        bundles = sorted(bundles, key=lambda x: x[0], reverse=True)
        
        scores = [bundle[0] for bundle in bundles]
        # Create a bar plot of the scores
        indices = np.arange(len(scores))
        plt.figure(figsize=(8, 4))
        plt.bar(indices, scores.cpu().numpy() if hasattr(scores, "cpu") else scores, color='skyblue')
        plt.xlabel("Belief Rank")
        plt.ylabel("Belief Score")
        plt.title(f"{prefix}Belief Scores for '{querry}'")
        
        # Plot n_cols RGB masks in a grid of the top n_rows candidates
        n_rows = min(n_rows, len(bundles))
        max_imgs = max([len(rgb_masks) for _, rgb_masks in bundles[:n_rows]])
        n_cols = min(n_cols, max_imgs)
        n_cols = max(n_cols, 2)  # Ensure at least one column
        #print(f"Visualizing {querry} with {n_rows} rows and {n_cols} columns of images.")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(19,12))
        fig.suptitle(f"{prefix}Belief Scores for '{querry}'", fontsize=16)
        for i, (score, rgb_masks)in enumerate(bundles[:n_rows]):
            axes[i, 0].set_title(f"{querry} {i} Score: {score:.2f}")
            for j in range(min(len(rgb_masks), n_cols)):
                axes[i, j].imshow(rgb_masks[j])
                axes[i, j].axis('off')
        fig.tight_layout()
        plt.show(block=blocking)
        #plt.pause(0.1)

    def vis_belief3D(self, querry):
        """
        Visualizes the belief scores of a given object in 3D using Open3D.
        Displays the point clouds and 3D bounding boxes in an Open3D visualizer.
        """
        if querry not in self.tracked_objects:
            raise ValueError(f"Object {querry} not found in tracked objects.")
        pcds = self.tracked_objects[querry]["pcds"]
        scores = self.tracked_objects[querry]["scores"]
        bboxes = self.tracked_objects[querry]["boxes"]

        # 2) Initialize the GUI App (only once per process)
        app = gui.Application.instance
        app.initialize()

        # 3) Create an O3DVisualizer window
        vis = o3d.visualization.O3DVisualizer(f"Belief Vis: {querry}", 1024, 768)
        vis.show_settings = True

        # 4) Add each point cloud and box to the scene
        for idx, pc in enumerate(pcds):
            vis.add_geometry(f"pcd_{idx}", pc)
        for idx, box in enumerate(bboxes):
            vis.add_geometry(f"box_{idx}", box)

        # 5) Annotate each box’s center with its score
        for box, score in zip(bboxes, scores):
            # get_center() works for AABB, OBB, and mesh
            center = box.get_center() if hasattr(box, "get_center") else np.asarray(box.vertices).mean(axis=0)
            vis.add_3d_label(center, f"{float(score):.3f}")

        # 6) Camera & run
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()

    def display(self):
        """
        Displays the tracked objects in a 3D visualizer using Open3D.
        """

        app = gui.Application.instance
        app.initialize()

        # 2) Create an O3DVisualizer window
        vis = o3d.visualization.O3DVisualizer("Final finding", 1024, 768)
        vis.show_settings = True

        # 3) Add a camera‐frame axis
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2,
            origin=[0, 0, 0]
        )
        vis.add_geometry("CameraFrame", camera_frame)

        # 4) For each query: get its point cloud + belief, add both geometry + label
        for q in self.tracked_objects.keys():
            pcd, belief = self.querry(q)
            # add the raw point cloud
            vis.add_geometry(f"pcd_{q}", pcd)
            # compute a label position (centroid of the cloud)
            pts = np.asarray(pcd.points)
            if pts.size:
                center = pts.mean(axis=0)
            else:
                center = np.array([0.0, 0.0, 0.0])
            # place a 3D text label of the belief
            vis.add_3d_label(center, f"{q}:{belief:.3f}")

        # 5) Finalize camera & run
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()


def test_VP(cap):
    vp = VisionPipe()
    for i in range(10):
        ret, rgb_img, depth_img = cap.read(return_depth=True)
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break
        I = cap.get_intrinsics()
        print(f"Frame {i}:")
        predictions = vp.update(rgb_img, depth_img, config["test_querys"], I)
        for object, prediction in predictions.items():
            print(f"{object=}")
            print(f"   {len(prediction['boxes'])=}, {len(prediction['pcds'])=}, {prediction['scores'].shape=}")
        print(f"\n\n")
    
        # for q in config["test_querys"]:
        #     vp.vis_belief3D(q)
    vp.vis_belief2D(querry=config["test_querys"][0], blocking=True, n_rows=5, n_cols=3)
    #vp.display()





if __name__ == "__main__":
    cap = RealSenseCamera()
    I = cap.get_intrinsics()
    ret, rgb_img, depth_img = cap.read(return_depth=True)
    if not ret:
        print("Error: Unable to read frame from the camera.")
        exit(1)

    print(f"\n\nTESTING VP")
    test_VP(cap)