config = {
    #----- VLM SETTINGS ----
    "gemini_model": "gemini-2.5-flash-lite-preview-06-17",
    "gemini_max_retries": 4,

    "owl_min_2d_box_side": 0,
    "owl_iou_2d_reduction": 0.3,
    "owlv2_discard_percentile": 0.75,
    "owlv2_sigmoid_gain": 1,

    "yolo_world_weights": "yolov8x-worldv2_best.pt", #path to yolo weights contained in core/ModelWeights use None to load deafault weights
    "backbone": "yoloworld", #choices are "yoloworld", "owlv2", "gemini"

    "sam2_model": "sam2.1_hiera_large.pt", #sam weights contained in core/ModelWeights
    "sam2_config": "sam2.1_hiera_l.yaml", #configes installed with sam


    #----- Belief update settings ----
    "remove_belief_threshold": -1.0, 
    "visible_portion_requirement": 0.75, #Fraction of a point cloud that must be considered visible for an object to count as “in view.”
    "occlusion_tol":0.1, #Tolerance margin for occlusion checks.
    "vlm_true_positive_rate": 0.6, #minimum confidence produced by bb backbone (must be > 0.5)
    "vlm_false_negative_rate": 0.4, #probibility used to decay a missing object that should've been detected (must be < 0.5)
    "change_in_pose_threshold": 0.0, #meters, minimum distance the camera must move to trigger a new update
    "pose_expire_time": 0.0, #seconds, how long to keep a prior pose in memory


    #----- Point cloud noise settings ----
    "min_3d_points": 100, #minimum number of 3D points in a candidate to be considered
    "voxel_size": 0.001, #1mm

    "statistical_outlier_removal": True,
    "statistical_nb_neighbors": 100,
    "statistical_std_ratio": 0.5,
    "radius_outlier_removal": False,
    "radius_radius": 0.01,
    "radius_nb_points": 50,
    "camera_min_range_m": 0.3,
    "camera_max_range_m": 3.0,

    #----- Point cloud matching settings ----
    "candidate_match_method": "mean_nn",
    "mean_nn_match_threshold": 0.02,
    "chamfer_match_threshold": 0.10,
    "hausdorff_match_threshold": 0.10,
    "fscore_match_threshold":0.2,
    "fscore_radius":0.01,
    "mahalanobis_match_threshold": 4.0,


    #----- Default Queries -----
    "test_querys": [
        "soda can",
        "wrench",
        "drill",
        "screwdriver",
        "tape measure",
        "Bolt",
        "BusBar",
        "InteriorScrew",
        "Nut",
        "OrangeCover",
        "Screw",
        "Screw Hole"
    ],
    #----- Expected Cameras -----
    "rs_name_spaces": [
        "/realsense/head",
        "/realsense/left_hand"
    ],


    #----- Visualization settings-----
    "base_frame": "pelvis",
    "vis_k": -1, #-1 for all otherwise display the top K candidates
    "save_figs": True
    


}