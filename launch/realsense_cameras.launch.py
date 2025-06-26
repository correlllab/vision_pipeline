#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments for parameterization
    declared_arguments = [
        DeclareLaunchArgument('enable_pointcloud', default_value='false'),
        DeclareLaunchArgument('enable_rgbd', default_value='false'),
        DeclareLaunchArgument('launch_head', default_value='true'),
        DeclareLaunchArgument('launch_left_hand', default_value='true'),
        DeclareLaunchArgument('launch_right_hand', default_value='true')
    ]

    # Launch configurations
    pointcloud = LaunchConfiguration('enable_pointcloud')
    rgbd = LaunchConfiguration('enable_rgbd')
    launch_head = LaunchConfiguration('launch_head')
    launch_left_hand = LaunchConfiguration('launch_left_hand')
    launch_right_hand = LaunchConfiguration('launch_right_hand')

    pkg_share = get_package_share_directory('realsense2_camera')
    rs_launch_file = os.path.join(pkg_share, 'launch', 'rs_launch.py')

    def make_camera(name, serial, width, height, fps, condition):
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
            condition=IfCondition(condition)
        )

    return LaunchDescription(declared_arguments + [
        make_camera('head', '_250122072330', width=424, height=240, fps=6, condition=launch_head),
        make_camera('left_hand', '_838212072778', width=320, height=240, fps=6, condition=launch_left_hand),
        make_camera('right_hand', '_926522071700', width=320, height=240, fps=6, condition=launch_right_hand),
    ])
