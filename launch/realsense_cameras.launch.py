#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import json

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vision_pipeline', 'config.json'))
config = json.load(open(config_path))
def generate_launch_description():
    # Declare launch arguments for parameterization
    declared_arguments = [
        DeclareLaunchArgument('enable_pointcloud', default_value='false'),
        DeclareLaunchArgument('enable_rgbd', default_value='false'),
    ]

    # Launch configurations
    pointcloud = LaunchConfiguration('enable_pointcloud')
    rgbd = LaunchConfiguration('enable_rgbd')

    pkg_share = get_package_share_directory('realsense2_camera')
    rs_launch_file = os.path.join(pkg_share, 'launch', 'rs_launch.py')

    def make_camera(name, serial, width, height, fps):
        return GroupAction(
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(rs_launch_file),
                    launch_arguments={
                        'camera_name': name,
                        'camera_namespace': 'realsense',
                        'serial_no': serial,
                        'rgb_camera.color_profile': f'{width}x{height}x{fps}',
                        'enable_color': 'true',
                        'enable_depth': 'true',
                        'pointcloud.enable': pointcloud,
                        'align_depth.enable': 'true',
                        'enable_rgbd': rgbd,
                        'enable_sync': 'true',
                    }.items()
                )
            ],
        )
    cams = []
    for camera_name, serial, frame in zip(config['rs_names'], config['rs_serials'], config['rs_frames']):
        cams.append(
            make_camera(
                camera_name,
                serial,
                width = config['rs_width'],
                height = config['rs_height'],
                fps = config['rs_fps'],
            )
        )
    return LaunchDescription(declared_arguments + cams)
