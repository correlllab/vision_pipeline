
import time
import numpy as np
import os
import sys
ros_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(ros_dir, "..")
utils_dir = os.path.join(parent_dir, "utils")
core_dir = os.path.join(parent_dir, "core")
fig_dir = os.path.join(parent_dir, 'figures')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if ros_dir not in sys.path:
    sys.path.insert(0, ros_dir)
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)
import json
config = json.load(open(os.path.join(parent_dir, "config.json")))

from ros_utils import msg_to_pcd
from behaviors import MainNode
from open3d.t.geometry import Metric, MetricParameters
import matplotlib
matplotlib.use('TkAgg') # Use the Tkinter backend
import matplotlib.pyplot as plt


def plot_errorbars(ax, trajectory_list, x):
    data = np.array(trajectory_list)
    n_trajectories, n_steps = data.shape
    mean_trajectory = np.mean(data, axis=0)
    min_trajectories = np.min(data, axis=0)
    max_trajectories = np.max(data, axis=0)
    for trajectory in trajectory_list:
        ax.plot(x, trajectory, color="grey", alpha=0.5, linewidth=1)
    ax.fill_between(x, min_trajectories, max_trajectories, color="lightblue", alpha=0.5)
    ax.plot(x,mean_trajectory, color="orange")
    



def distance_filter(clouds_msg, probs, thresh = 0.85):
    pcds = [msg_to_pcd(cloud) for cloud in clouds_msg]
    cloud_points = [np.asarray(pcd.to_legacy().points) for pcd in pcds]
    
    final_points, final_probs, final_o3d = [], [], []
    for i, (points, prob, o3d_pcd) in enumerate(zip(cloud_points, probs, pcds)):
        x_y_distances = np.linalg.norm(points[:,:2], axis=1)
        avg_x_y_distance = np.mean(x_y_distances)
        mean_y = np.mean(points, axis=1)[1]
        if avg_x_y_distance < thresh and mean_y > 0.1:
            final_points.append(points)
            final_probs.append(prob)
            final_o3d.append(o3d_pcd)
            # print(f"keeping {i}: {prob} {avg_x_y_distance}m")
        else:
            pass
            # print(f"discarding {i}: {prob} {avg_x_y_distance}m")
    # print(f"kept {len(final_points)}/{len(cloud_points)}")
    return final_points, final_probs, final_o3d

def get_point_point(points):
    highest_z = np.max(points[:,2])
    x_y_distances = np.linalg.norm(points[:,:2], axis=1)
    min_dist_idx = np.argmin(x_y_distances)
    closest_point = points[min_dist_idx]
    x = closest_point[0]
    y = closest_point[1]
    return x, y, highest_z
def main():
    print("entered main")
    node = MainNode()
    node.go_home()
    node.reset_beliefs()
    node.update_head()
    


    sufficient_prob = 0.95
        
    objects = config["test_querys"]
    last_input = ""
    #loop to find objects over and over
    data = {}
    metric_params = MetricParameters()
    while last_input != "q":
        int_str_mapping = {str(i): obj for i, obj in enumerate(objects)}
        print(int_str_mapping)
        last_input = input("Enter the index of the object to query or 'q' to quit: ")
        if last_input == 'q':
            print("Exiting...")
            return
        goal_object = objects[int(last_input)]
        data[goal_object] = {"prob_trajectories":[], "n_point_trajectories":[]}
        query_sucess = False
        #make sure we have a starting point
        while not query_sucess:
            query = node.query_objects(goal_object, threshold=0.0)
            query_sucess = query.success

        current_points, current_probs, current_o3d_clouds = distance_filter(query.clouds, query.probabilities)
        sample_distances = np.arange(0.7, 0.2, -0.05)
        k_max = len(sample_distances)
        print(f"\n\n\nbeginging {goal_object}")
        print(f"{max(current_probs)=}, {min(current_probs)=}, {len(current_points)= }, {len(current_o3d_clouds)=}, {len(current_probs)=}")
        for i, (cur_point_set, cur_prob, cur_cloud) in enumerate(zip(current_points, current_probs, current_o3d_clouds)):
            node.go_home()
            node.reset_beliefs()
            node.update_head()
            centroid = np.mean(cur_point_set, axis=0)
            data[goal_object]["prob_trajectories"].append([])
            data[goal_object]["n_point_trajectories"].append([])

            if cur_prob > sufficient_prob:
                print(f"keeping {centroid=} {cur_prob=}")
                continue
            print(f"pointing at {i+1}/{len(current_points)} {centroid=} {cur_prob=} d={np.linalg.norm(centroid)}")

            for i, height in enumerate(sample_distances):
                print(f"height {i+1}/{len(sample_distances)}")
                node.point_camera(centroid[0], centroid[1], centroid[2], height=height)
                node.update_hand()

                query_sucess = False
                #make sure we have a starting point
                query = None
                while not query_sucess:
                    query = node.query_objects(goal_object, threshold=0.0)
                    query_sucess = query.success
                new_points, new_probs, new_o3d_clouds = distance_filter(query.clouds, query.probabilities)
                print(f"{max(new_probs)=}, {min(new_probs)=}, {len(new_points)= }, {len(new_o3d_clouds)=} {len(new_probs)=}")
                distances = [cur_cloud.compute_metrics(new_cloud, [Metric.ChamferDistance], metric_params) for new_cloud in new_o3d_clouds]
                min_distance_idx = np.argmin(np.array(distances))
                print(f"{min(distances)=}")
                new_prob = new_probs[min_distance_idx]
                n_points = new_points[min_distance_idx].shape[0]
                data[goal_object]["prob_trajectories"][-1].append(new_prob)
                data[goal_object]["n_point_trajectories"][-1].append(n_points)
                
        print("finished updating, plotting")
        graphs = [
            (np.arange(k_max), "Step (k)", data[goal_object]["prob_trajectories"], "belief", False, True),
            (np.arange(k_max), "Step (k)", data[goal_object]["n_point_trajectories"], "n points", False, False),
            (sample_distances, "Camera Distance (m)", data[goal_object]["prob_trajectories"], "belief", True, True),
            (sample_distances, "Camera Distance (m)", data[goal_object]["n_point_trajectories"], "n points", True, False),
            ]

        print("setting up graphs")
        cols = len(graphs)
        fig, axes = plt.subplots(nrows = 1, ncols=cols, figsize=(20,20))
        axes = axes.flatten()

        print("Populating graphs")
        for i, (x, x_name, y, y_name, invert_x, y_range_01) in enumerate(graphs):
            ax = axes[i]
            ax.set_title(f"{x_name} vs {y_name}")
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            print(f"{x=}")
            print(f"{y=}")

            plot_errorbars(ax, y, x)
            if invert_x:
                ax.invert_xaxis()
            if y_range_01:
                ax.set_ylim(0,1.1)
            else:
                ax.set_ylim(bottom=0)

        print("saving graphs")

        plt.tight_layout()
        plt.savefig(f'{goal_object}.png') 
        plt.show()

        if len(current_points) == 0:
            print("no pointclouds remain")
            return
        if len(current_probs) == 0:
            print("no probs remain")
            return
            
    
    

if __name__ == '__main__':
    main()