#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import json

config_path = os.path.join(get_package_share_directory('vision_pipeline'), 'config.json')
config = json.load(open(config_path))
def generate_launch_description():

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
                        'depth_module.depth_profile':  f'{width}x{height}x{fps}',
                        'enable_color': 'true',
                        'enable_depth': 'true',
                        'pointcloud.enable': 'true',
                        'align_depth.enable': 'true',
                        'initial_reset': 'true',
                        'enable_sync': 'true',
                        'decimation_filter.enable': 'true',
                        'enable_rgbd': 'false',
                        'enable_motion': 'false',
                        'enable_infra' : 'false',
                        'enable_infra1': 'false',
                        'enable_infra2': 'false',
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
    return LaunchDescription(cams)
