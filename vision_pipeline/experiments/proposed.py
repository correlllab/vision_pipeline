
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


def distance_filter(clouds_msg, probs, thresh = 0.6):
    pcds = [msg_to_pcd(cloud) for cloud in clouds_msg]
    cloud_points = [np.asarray(pcd.to_legacy().points) for pcd in pcds]
    
    final_points, final_probs = [], []
    for points, prob in zip(cloud_points, probs):
        x_y_distances = np.linalg.norm(points[:,:2], axis=1)
        x_y_distances *= np.sign(points[:, 1])
        avg_x_y_distance = np.mean(x_y_distances)
        print(f"{avg_x_y_distance=}")
        if avg_x_y_distance < thresh:
            final_points.append(points)
            final_probs.append(prob)
    return final_points, final_probs

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
    node.update_head()


    sufficient_prob = 1.01
        
    objects = config["test_querys"]
    last_input = ""
    #loop to find objects over and over
    while last_input != "q":
        int_str_mapping = {str(i): obj for i, obj in enumerate(objects)}
        print(int_str_mapping)
        last_input = input("Enter the index of the object to query or 'q' to quit: ")
        if last_input == 'q':
            print("Exiting...")
            return
        goal_object = objects[int(last_input)]
        node.update_head()
        query_sucess = False
        #make sure we have a starting point
        while not query_sucess:
            query = node.query_objects(goal_object, threshold=0.0)
            query_sucess = query.success

        points, probs = distance_filter(query.clouds, query.probabilities)

        print(f"{max(probs)=}")
        while max(probs) < sufficient_prob and len(points) > 0 and len(probs) > 0:
            for point_set in points:
                centroid = np.mean(point_set, axis=0)
                node.point_camera(centroid[0], centroid[1], centroid[2])
                node.update_hand()
            query_sucess = False
            #make sure we have a starting point
            while not query_sucess:
                query = node.query_objects(goal_object, threshold=0.0)
                query_sucess = query.success
            points, probs = distance_filter(query.clouds, query.probabilities)
            print(f"{max(probs)=}")

        if len(points) == 0:
            print("no pointclouds remain")
            return
        if len(probs) == 0:
            print("no probs remain")
            return
            

        max_idx = np.argmax(probs)
        candidate_mask = np.array(probs) > sufficient_prob
        n_valid = np.sum(candidate_mask)
        for i, valid in enumerate(candidate_mask):
            if not valid:
                continue
            max_prob_points = points[i]

            x,y,z = get_point_point(max_prob_points)
            
            node.point_finger(x, y, z)
            input(f"continue? {i+1}/{n_valid}")
    
    

if __name__ == '__main__':
    main()