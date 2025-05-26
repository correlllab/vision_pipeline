from VisionPipeline.vision_pipeline.FoundationModels import OWLv2, SAM2_PC, display_owl, display_sam2
import open3d as o3d
from vision_pipeline.utils import iou_3d, pose_to_matrix, matrix_to_pose, in_image
import torch
from open3d.visualization import gui, rendering
import numpy as np
import json
import os
from VisionPipeline.vision_pipeline.CameraInterfaces import RealSenseCamera, RealSenseSubscriber
import rclpy
import open3d as o3d
_script_dir = os.path.dirname(os.path.realpath(__file__))
_config_path = os.path.join(_script_dir, 'config.json')
config = json.load(open(_config_path, 'r'))


class VisionPipe:
    def __init__(self):
        self.owv2 = OWLv2()
        self.sam2 = SAM2_PC()
        self.tracked_objects = {}
    def update(self, rgb_img, depth_img, querries, I, obs_pose=[0,0,0,0,0,0], debug = False):
        predictions_2d = self.owv2.predict(rgb_img, querries, debug=debug)
        if debug:
            display_owl(rgb_img, predictions_2d)
        predictions_3d = {}
        transformation_matrix = pose_to_matrix(obs_pose)
        for object, prediction_2d in predictions_2d.items():
            pcds, box_3d, scores = self.sam2.predict(rgb_img, depth_img, prediction_2d["boxes"], prediction_2d["scores"], I, debug=debug)
            pcds = [pcd.transform(transformation_matrix) for pcd in pcds]
            if debug:
                display_sam2(pcds, box_3d, scores, window_prefix=f"{object} ")
            predictions_3d[object] = {"boxes": box_3d, "scores": scores, "pcds": pcds}
            if debug:
                print(f"{object=}")
                print(f"   {len(predictions_3d[object]['boxes'])=}, {len(predictions_3d[object]['pcds'])=}, {predictions_3d[object]['scores'].shape=}")

        self.update_tracked_objects(predictions_3d, obs_pose, I, debug=debug)
        return self.tracked_objects

    def update_tracked_objects(self, predictions_3d, obs_pose, I, debug = False):
        for object, predictions in predictions_3d.items():
            if object not in self.tracked_objects:
                self.tracked_objects[object] = predictions_3d[object]
            else:
                updated = [False for i in range(len(self.tracked_objects[object]["boxes"]))]
                for candidate_box, candidate_score, candidate_pcd in zip(predictions["boxes"], predictions["scores"], predictions["pcds"]):
                    ious = [iou_3d(candidate_box, box) for box in self.tracked_objects[object]["boxes"]]
                    max_iou = max(ious)
                    if max_iou < config["iou_3d"]:
                        self.tracked_objects[object]["boxes"].append(candidate_box)
                        self.tracked_objects[object]["pcds"].append(candidate_pcd)

                        # your existing scores (shape: torch.Size([32]))
                        scores = self.tracked_objects[object]['scores']

                        # make it a 1-element tensor
                        new_score = candidate_score.unsqueeze(0)   # shape: [1]

                        # concatenate along dim=0
                        updated_scores = torch.cat([scores, new_score], dim=0)  # shape: [33]

                        # store it back
                        self.tracked_objects[object]['scores'] = updated_scores
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

                        # existing belief and new observation
                        b = self.tracked_objects[object]["scores"][match_idx]
                        x = candidate_score

                        # weighted‐power update
                        num = b * x
                        den = num + ((1-b) * (1-x))

                        self.tracked_objects[object]["scores"][match_idx] = num / den
                        updated[match_idx] = True
                """for i, (tracked_score, pcd, obj_updated) in enumerate(zip(self.tracked_objects[object]["scores"], self.tracked_objects[object]["pcds"], updated)):
                    if obj_updated:
                        continue
                    centroid = pcd.get_center()

                    if not in_image(centroid, obs_pose, I):
                       continue
                    p_fn = config["false_negative_rate"]
                    b    = self.tracked_objects[object]["scores"][i]
                    w    = config["new_sample_weight"]

                    num = b**w * (p_fn)**(1-w)
                    den = num + (1-b)**w * (1-p_fn)**(1-w)
                    new_score = num/den

                    self.tracked_objects[object]["scores"][i] = new_score"""



        self.remove_low_belief_objects()
    def remove_low_belief_objects(self):
        for object, prediction in self.tracked_objects.items():
            #Remove objects with belief score < config["belief_threshold"]
            #print(f"{object=}")
            #print(f"   {len(prediction['boxes'])=}, {len(prediction['pcds'])=}, {prediction['scores'].shape=}")
            mask = prediction["scores"] > config["belief_threshold"]
            #print(f"{mask.shape=}")
            prediction["boxes"] = [prediction["boxes"][i] for i in range(len(mask)) if mask[i]]
            prediction["pcds"] = [prediction["pcds"][i] for i in range(len(mask)) if mask[i]]
            prediction["scores"] = prediction["scores"][mask]
    def querry(self, querry):
        if querry in self.tracked_objects:
            candiates = self.tracked_objects[querry]
            #print(f"{candiates=}")
            argmax1, maxval1 = max(enumerate(candiates["scores"]), key=lambda pair: pair[1])
            top_candiate = candiates["pcds"][argmax1]
            return top_candiate, maxval1
        raise ValueError(f"Object {querry} not found in tracked objects.")

    def vis_belief(self, querry):
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
        #     vp.vis_belief(q)
    vp.display()





if __name__ == "__main__":
    cap = RealSenseCamera()
    I = cap.get_intrinsics()
    ret, rgb_img, depth_img = cap.read(return_depth=True)
    if not ret:
        print("Error: Unable to read frame from the camera.")
        exit(1)

    print(f"\n\nTESTING VP")
    test_VP(cap)