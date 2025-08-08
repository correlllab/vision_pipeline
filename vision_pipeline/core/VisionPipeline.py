import open3d as o3d
import numpy as np
import json
import os
import time

import sys

from open3d.visualization import gui, rendering
from open3d.t.geometry import Metric as o3dMetrics
from open3d.t.geometry import MetricParameters as MetricParameters


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

    
        self.match_method = None
        self.match_threshold = None
        self.metric_parameters = MetricParameters(n_sampled_points=100, fscore_radius=[config["fscore_radius"]]) #n_sampled_points used for meshes. ignored for pointclouds
        if config["candidate_match_method"] == "iou":
            self.match_method = "iou"
            self.match_threshold = config["iou3d_match_threshold"]
        elif config["candidate_match_method"] == "chamfer":
            self.match_method = [o3dMetrics.ChamferDistance]
            self.match_threshold = config["chamfer_match_threshold"]            
        elif config["candidate_match_method"] == "hausdorff":
            self.match_method = [o3dMetrics.HausdorffDistance]
            self.match_threshold = config["hausdorff_match_threshold"]            
        elif config["candidate_match_method"] == "fscore":
            self.match_method = [o3dMetrics.FScore]
            self.match_threshold = config["fscore_match_threshold"]
        else:
            raise ValueError(f"Unknown match method {config['candidate_match_method']=}. Please choose 'iou','chamfer','hausdorff' or 'fscore'.")


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
        
        out_str = "successfully_updated"
        if debug:
            out_str+="\n\n\n"

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
            for object_str in candidates_3d:
                print(f"[VisionPipe get_candidates]{object_str=}")
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
        if new_prob >= 1:
            new_prob = 0.99
        return new_prob

    def merge_candidate(self, candidate_box, candidate_prob, candidate_pcd, candidate_rgb_mask, candidate_depth_mask, obj_str, n_considered):
        #calculate the 3D IoU with all tracked objects
        distances = None
        if self.match_method == "iou":
            distances = [1 - iou_3d(candidate_box, box) for box in self.tracked_objects[obj_str]["boxes"][:n_considered]]
        else:
            distances = [candidate_pcd.compute_metrics(tracked_pcd, self.match_method, self.metric_parameters).item() for tracked_pcd in self.tracked_objects[obj_str]["pcds"][:n_considered]]
        if config["candidate_match_method"] == "fscore":
            distances = [1-dist for dist in distances]

        min_dist = float("inf")
        if len(distances) > 0:
            min_dist = min(distances)
       
        match_idx = None
        #if the distance is more than the threshold, its a new object
        if min_dist > self.match_threshold:
            self.tracked_objects[obj_str]["probs"].append(candidate_prob)
            self.tracked_objects[obj_str]["boxes"].append(candidate_box)
            self.tracked_objects[obj_str]["pcds"].append(candidate_pcd)
            self.tracked_objects[obj_str]["rgb_masks"].append([candidate_rgb_mask])
            self.tracked_objects[obj_str]["depth_masks"].append([candidate_depth_mask])

            next_idx = None
            try:
                if len(self.tracked_objects[obj_str]["names"]) == 0:
                    next_idx = 1
                else:
                    previous_names = self.tracked_objects[obj_str]["names"][-1]
                    next_idx = int(previous_names.split("_")[-1]) + 1
            except Exception as e:
                print(f"[VisionPipe merge_candidate] Exception {e} with {self.tracked_objects[obj_str]['names']=}")
                next_idx = 1
                
            self.tracked_objects[obj_str]["names"].append(f"{obj_str}_{next_idx}")

        # if the distance is less than the threshold its an alreadly tracked object
        else:
            match_idx = distances.index(min_dist)
            pcd = self.tracked_objects[obj_str]["pcds"][match_idx] + candidate_pcd
            pcd = pcd.voxel_down_sample(voxel_size=config["voxel_size"])
            # Apply statistical outlier removal to denoise the point cloud
            if config["statistical_outlier_removal"]:
                pcd, ind = pcd.remove_statistical_outliers(nb_neighbors=config["statistical_nb_neighbors"], std_ratio=config["statistical_std_ratio"])
            if config["radius_outlier_removal"]:
                pcd, ind = pcd.remove_radius_outliers(nb_points=config["radius_nb_points"], search_radius=config["radius_radius"])
            self.tracked_objects[obj_str]["pcds"][match_idx] = pcd
            self.tracked_objects[obj_str]["boxes"][match_idx] = pcd.get_axis_aligned_bounding_box()
            self.tracked_objects[obj_str]["rgb_masks"][match_idx].append(candidate_rgb_mask)
            self.tracked_objects[obj_str]["depth_masks"][match_idx].append(candidate_depth_mask)
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
            #for each candidate either merge with an existing tracked object or add as a new tracked object
            for i, (candidate_box, candidate_prob, candidate_pcd, candidate_rgb_mask, candidate_depth_mask) in enumerate(zip(candidates["boxes"], candidates["probs"], candidates["pcds"], candidates["rgb_masks"], candidates["depth_masks"])):
                merge_idx = self.merge_candidate(candidate_box, candidate_prob, candidate_pcd, candidate_rgb_mask, candidate_depth_mask, obj_str, n_preupdated_tracked_objects)
                if merge_idx is not None:
                    updated[merge_idx] = True
                if debug:
                    print(f"[VisionPipe update_tracked_objects]{obj_str=}, candidate {i} matched to tracked object {merge_idx}")
            if debug:
                print("\n\n")
            # If the object was not updated but it should've been, we need to update its belief score
            for i, (tracked_prob, pcd, obj_updated) in enumerate(zip(self.tracked_objects[obj_str]["probs"], self.tracked_objects[obj_str]["pcds"], updated)):
                if obj_updated:
                    continue
                centroid = pcd.get_center().numpy()

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
        for obj_str, curr_prediction in self.tracked_objects.items():
            mask = [prob > config["remove_belief_threshold"] for prob in curr_prediction["probs"]]

            new_prediction = {}
            pcds = []
            boxes = []
            probs = []
            rgb_masks = []
            depth_masks = []
            names = []
            for i, b in enumerate(mask):
                if b:
                    pcds.append(curr_prediction["pcds"][i])
                    boxes.append(curr_prediction["boxes"][i])
                    probs.append(curr_prediction["probs"][i])
                    rgb_masks.append(curr_prediction["rgb_masks"][i])
                    depth_masks.append(curr_prediction["depth_masks"][i])
                    names.append(curr_prediction["names"][i])
            new_prediction["pcds"] = pcds
            new_prediction["boxes"] = boxes
            new_prediction["probs"] = probs
            new_prediction["rgb_masks"] = rgb_masks
            new_prediction["depth_masks"] = depth_masks
            new_prediction["names"] = names

            self.tracked_objects[obj_str] = new_prediction
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


def main():
    import threading, time, traceback
    from realsense_devices import RealSenseCamera



    queries = config["test_querys"]


    # Display Variables 
    running = True
    lock = threading.Lock()
    latest_pcs = {}
    latest_bbs = {}
    latest_probs = {}
    TICK_PERIOD = 0.05
    obs_pose = [0,0,0,0,0,0]
    display_world = True
    



    # Open3D GUI setup 
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("RealSense 3D Viewer", 1024, 768)
    vis.show_settings = True
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry("world_axis", coord_frame)

    if display_world:
        WORLD_MAX_POINTS = 300_000
        positions_np = np.zeros((WORLD_MAX_POINTS, 3), dtype=np.float32)
        colors_np = np.zeros((WORLD_MAX_POINTS, 3), dtype=np.float32)
        device = o3d.core.Device("CPU:0")
        world_pc = o3d.t.geometry.PointCloud()
        world_pc.point["positions"] = o3d.core.Tensor(positions_np, o3d.core.float32, device)
        world_pc.point["colors"]    = o3d.core.Tensor(colors_np,   o3d.core.float32, device)
        vis.add_geometry("world_pc", world_pc)
        latest_vertices = None
        latest_colors = None


    vis.reset_camera_to_default()
    app.add_window(vis)

    last_frame = set()
    do_debug = False
    # ---- RealSense grabber thread (no GUI calls here) ----
    def grabber():
        nonlocal latest_pcs, latest_bbs, latest_probs, running
        if display_world:
            nonlocal latest_vertices, latest_colors
        cam = None
        vp = None
        try:
            vp = VisionPipe()
            cam = RealSenseCamera()
            # short warmup
            for _ in range(10):
                data = cam.get_data()
                time.sleep(0.02)
            fx=data['color_intrinsics'].fx
            fy=data['color_intrinsics'].fy
            cx=data['color_intrinsics'].ppx
            cy=data['color_intrinsics'].ppy
            width = data["color_intrinsics"].width
            height = data["color_intrinsics"].height
            intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}
            while running:
                data = cam.get_data()  # make sure this has a short internal timeout
                rgb_img = data["color_img"]
                depth_img = data["depth_img"] * cam.depth_scale
                success, message = vp.update(rgb_img, depth_img, queries, intrinsics, obs_pose, time_stamp=time.time(), debug=do_debug)
                # print(f"Update success: {success}, message: {message}")

                tmp_latest_bbs = {}
                tmp_latest_pcs = {}
                tmp_latest_probs = {}
                for obj in vp.tracked_objects:
                    tmp_latest_bbs[obj] = vp.tracked_objects[obj]["boxes"]
                    tmp_latest_pcs[obj] = vp.tracked_objects[obj]["pcds"]
                    tmp_latest_probs[obj] = vp.tracked_objects[obj]["probs"]
          


                with lock:
                    latest_pcs = tmp_latest_pcs
                    latest_bbs = tmp_latest_bbs
                    latest_probs = tmp_latest_probs
                    if display_world:
                        latest_vertices = data["vertices"]
                        latest_colors = data["colors"]

                # tiny nap to avoid maxing a CPU
                time.sleep(0.1)
        except Exception as e:
            print("Grabber error:", e)
            traceback.print_exc()
        finally:
            if cam:
                try: cam.stop()
                except Exception: pass

    def pump():
        pcs = None
        bbs = None
        with lock:
            pcs   = {k: v[:] for k, v in latest_pcs.items()}
            bbs   = {k: v[:] for k, v in latest_bbs.items()}
            probs = {k: v[:] for k, v in latest_probs.items()}
            if display_world:
                verts = latest_vertices
                colors = latest_colors

        if not pcs or not bbs or not probs:
            return
        
        flags = (rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG)
        nonlocal last_frame
        this_frame = set()

        vis.clear_3d_labels()
        for q in queries:
            if q not in pcs or q not in bbs:
                continue
            q_pcs = pcs[q]
            q_bboxs = bbs[q]
            q_probs = probs[q]
            assert len(q_pcs) == len(q_bboxs)
            assert len(q_pcs) == len(q_probs)

            for i in range(len(q_pcs)):
                pcd = q_pcs[i]
                bbox = q_bboxs[i]
                prob = q_probs[i]

                # Point cloud
                name_pcd = f"{q.replace(' ', '_')}_pcd_{i}"
                if name_pcd in last_frame:
                    vis.remove_geometry(name_pcd)
                vis.add_geometry(name_pcd, pcd) 
                this_frame.add(name_pcd)
                

                # # Bounding box
                name_bbox = f"{q.replace(' ', '_')}_bbox_{i}"
                if name_bbox in last_frame:
                    vis.remove_geometry(name_bbox)
                legacy_bbox = bbox.to_legacy()
                legacy_bbox.color = (1-prob, prob, 0)
                vis.add_geometry(name_bbox, legacy_bbox)
                this_frame.add(name_bbox)

                vis.add_3d_label(pcd.get_center().numpy(), f"{q} {prob:.2f}")


        for stale_geometry in (last_frame - this_frame):
            vis.remove_geometry(stale_geometry)
            
        last_frame = this_frame

        if display_world:
            n = min(len(verts), WORLD_MAX_POINTS)
            if n > 0:
                # slice-assign into preallocated tensors (no reallocation)
                world_pc.point["positions"][:n] = o3d.core.Tensor(verts[:n],  o3d.core.float32, device)
                world_pc.point["colors"][:n]    =  o3d.core.Tensor(colors[:n],   o3d.core.float32, device)
            if n < WORLD_MAX_POINTS:
                # avoid NaNs; fill the tail with zeros
                world_pc.point["positions"][n:] = 0
                world_pc.point["colors"][n:]    = 0
            flags = (rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG)
            vis.update_geometry("world_pc", world_pc, flags)
            vis.post_redraw()

        vis.post_redraw()

    def ticker():
        while running:
            try:
                app.post_to_main_thread(vis, pump)
            except Exception:
                pass
            time.sleep(TICK_PERIOD)

    def on_close():
        nonlocal running
        running = False
        return True
    vis.set_on_close(on_close)

    # Start threads
    th_grab = threading.Thread(target=grabber, daemon=True)
    th_tick = threading.Thread(target=ticker,  daemon=True)
    th_grab.start(); th_tick.start()

    try:
        app.run()
    finally:
        # Stop threads and close plot
        running = False
        th_grab.join(timeout=2)
        th_tick.join(timeout=2)

if __name__ == "__main__":
    main()