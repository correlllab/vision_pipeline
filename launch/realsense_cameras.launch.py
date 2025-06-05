#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


Width = '640'
Height = '480'
fps = '30'
pointcloud_enable = 'true'


def generate_launch_description():
    # Path to the upstream rs_launch.py
    pkg_share = get_package_share_directory('realsense2_camera')
    rs_launch_file = os.path.join(pkg_share, 'launch', 'rs_launch.py')

    # Launch first camera: head
    head_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch_file),
        launch_arguments={
            'camera_name': 'head',
            'camera_namespace': 'realsense',
            'serial_no': '_250122072330',
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': pointcloud_enable,
            'align_depth.enable': 'true',
            'enable_rgbd': "true",
            "enable_sync": "true",
            'color_width':      Width,
            'color_height':     Height,
            'color_fps':        fps,
            'depth_width':      Width,
            'depth_height':     Height,
            'depth_fps':        fps,


        }.items(),
    )

    # Launch second camera: left_hand
    left_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch_file),
        launch_arguments={
            'camera_name': 'left_hand',
            'camera_namespace': 'realsense',
            'serial_no': '_838212072778',
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': pointcloud_enable,
            'align_depth.enable': 'true',
            'enable_rgbd': "true",
            "enable_sync": "true",
            'color_width':      Width,
            'color_height':     Height,
            'color_fps':        fps,
            'depth_width':      Width,
            'depth_height':     Height,
            'depth_fps':        fps,

        }.items(),
    )


    # Launch third camera: right_hand
    right_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch_file),
        launch_arguments={
            'camera_name': 'right_hand',
            'camera_namespace': 'realsense',
            'serial_no': '_926522071700',
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': pointcloud_enable,
            'align_depth.enable': 'true',
            'enable_rgbd': "true",
            "enable_sync": "true",
            'color_width':      Width,
            'color_height':     Height,
            'color_fps':        fps,
            'depth_width':      Width,
            'depth_height':     Height,
            'depth_fps':        fps,
        }.items(),
    )

    return LaunchDescription([
        head_cam,
        #left_cam,
	    #right_cam,
    ])
