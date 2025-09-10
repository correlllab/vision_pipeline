
import time

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

def main():
    print("entered main")
    node = MainNode()
        
    objects = config["test_querys"]
    last_input = ""
    while last_input != "q":
        int_str_mapping = {str(i): obj for i, obj in enumerate(objects)}
        print(int_str_mapping)
        last_input = input("Enter the index of the object to query or 'q' to quit: ")
        if last_input == 'q':
            print("Exiting...")
            return
        goal_object = objects[int(last_input)]
        success = False
        max_tries = 5
        tries = 0
        query = None
        while success == False and tries < max_tries:
            print("sending querry")
            query = node.query_objects(goal_object)
            success = query.result
            print(f"Query status: {query.message}")
            tries += 1
            if not success:
                time.sleep(5)
        if not success:
            print("Failed to query tracked objects after maximum tries.")
            continue
        print(f"Query result: {query.result}, message: {query.message}")
        if not query.result or query.prob <= 0:
            continue
        pcd = msg_to_pcd(query.cloud)
        
        center = pcd.to_legacy().get_center()
        print(center)
        node.point_at(center[0], center[1], center[2])
    
    

if __name__ == '__main__':
    main()