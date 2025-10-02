
import time
import numpy as np
import os
import sys
import random
experiment_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(experiment_dir, "..")
utils_dir = os.path.join(parent_dir, "utils")
core_dir = os.path.join(parent_dir, "core")
fig_dir = os.path.join(parent_dir, 'figures')
ros_dir = os.path.join(parent_dir, "ROS")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from config import config
if ros_dir not in sys.path:
    sys.path.insert(0, ros_dir)
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)
if experiment_dir not in sys.path:
    sys.path.insert(0, experiment_dir)
import json


from ros_utils import msg_to_pcd
from behaviors import BehaviorNode
from open3d.t.geometry import Metric, MetricParameters
import matplotlib
matplotlib.use('TkAgg') # Use the Tkinter backend
import matplotlib.pyplot as plt


csv_path = os.path.join(experiment_dir, "trajectories.csv")
def save_trajectory(
    obj_type, 
    distance_history, 
    belief_history, 
    n_points_history,
    tp
):
    pc_match_method = "mean_nn"
    # Create a DataFrame for the new row
    new_row = pd.DataFrame([{
        "obj type": obj_type,
        "pc match method": pc_match_method,
        "belief history": json.dumps(belief_history),
        "n_points history": json.dumps(n_points_history),
        "true positive":tp
    }])

    # Append to CSV if it exists, otherwise create it
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(csv_path, index=False)

    

def distance_filter(clouds_msg, probs, names, thresh = 0.85):
    cloud_points = [np.asarray(msg_to_pcd(cloud).to_legacy().points) for cloud in clouds_msg]
    final_points, final_probs, final_msgs, final_names = [], [], [], []
    for i, (points, prob, o3d_pcd, name) in enumerate(zip(cloud_points, probs, clouds_msg, names)):
        x_y_distances = np.linalg.norm(points[:,:2], axis=1)
        avg_x_y_distance = np.mean(x_y_distances)
        mean_y = np.mean(points, axis=0)[1]
        print(f"{avg_x_y_distance}, {mean_y=}")

        # print(f"{points.shape=} {np.mean(points, axis=0).shape=} {np.mean(points, axis=1).shape=}")
        if avg_x_y_distance < thresh and mean_y > -0.1:
            final_points.append(points)
            final_probs.append(prob)
            final_msgs.append(o3d_pcd)
            final_names.append(name)
            # print(f"keeping {i}: {prob} {avg_x_y_distance}m")
        else:
            pass
            # print(f"discarding {i}: {prob} {avg_x_y_distance}m")
    # print(f"kept {len(final_points)}/{len(cloud_points)}")
    return final_points, final_probs, final_msgs, final_names

def main():
    #create behavior node
    print("creating behavior node")
    node = BehaviorNode()
    print("entered main")

    n = 1
    goal_object = "BusBar" #OrangeCover" #"BusBar" #"Nut" #"Screw"

    for i in range(n):
        node.forget_everything()
        node.update_head()
        query_sucess = False
        #make sure we have a starting point
        while not query_sucess:
            query = node.query_objects(goal_object, threshold=0.0)
            query_sucess = query.success

        #set up experiment params
        sample_distances = np.arange(0.75, 0.3, -0.05)
        print(f"\n\n\nbegining {goal_object}")
        initial_points, initial_probs, initial_msgs, initial_names = distance_filter(query.clouds, query.probabilities, query.names)
        print(f"{max(initial_probs)=}, {min(initial_probs)=}, {len(initial_points)= }, {len(initial_names)=}, {len(initial_probs)=}")
        initial_candidates = list(zip(initial_points, initial_probs, initial_names, initial_msgs))
        print(initial_names)
        cur_points, cur_prob, cur_name, cur_msg = random.choice(initial_candidates)
        for k in range(10):
            cur_points, cur_prob, cur_name, cur_msg = random.choice(initial_candidates)
            print(f"{cur_name=}")
        node.publish_pointcloud(cur_msg)

        #calculate viewing point
        centroid = np.mean(cur_points, axis=0)
        centroid_dist = np.linalg.norm(cur_points-centroid, axis=1)
        closest_idx = np.argmin(centroid_dist)
        view_point = cur_points[closest_idx]

        print(f"\n\n\npointing at {view_point=} {cur_prob=} {cur_name=} d={np.linalg.norm(view_point):.2f}")

        #take k1 .... kn
        prob_trajectory = [cur_prob]
        n_point_trajectory = [cur_points.shape[0]]
        last_prob = prob_trajectory[-1]
        last_n_points = n_point_trajectory[-1]
        for j, height in enumerate(sample_distances):
            print(f"\n\nheight {j+1}/{len(sample_distances)} {i+1}/{n} {cur_name=}")
            node.point_camera(view_point[0], view_point[1], view_point[2], height=height)
            time.sleep(1)
            node.update_hand()

            query_sucess = False
            #make sure we have a starting point
            query = None
            while not query_sucess:
                query = node.query_objects(goal_object, threshold=0.0, specific_name=cur_name)
                query_sucess = query.success
            new_prob = query.probabilities[0]
            new_cloud = query.clouds[0]
            new_pcd = msg_to_pcd(new_cloud)
            new_points = np.asarray(new_pcd.to_legacy().points)
            n_points = new_points.shape[0]

            
            node.publish_pointcloud(new_cloud)
            print(f"prob:{last_prob:.2f}->{new_prob:.2f}, points:{last_n_points}->{n_points}")
            last_prob = new_prob
            last_n_points = n_points
            prob_trajectory.append(new_prob)
            n_point_trajectory.append(n_points)

        print(f"prob:{cur_prob:.2f}->{new_prob:.2f}, points:{cur_points.shape[0]}->{n_points}")
        tp_inp = input("true positive y/n: ")
        tp = None
        if tp_inp == "n":
            tp = False
        else:
            tp = True
        print(f"{tp_inp=} {tp=}")
        save_trajectory(
            obj_type=goal_object,
            distance_history=list(sample_distances),
            belief_history=prob_trajectory,
            n_points_history=n_point_trajectory,
            tp=tp
        )
    
    

if __name__ == '__main__':
    main()