#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


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
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rs_launch_file),
            launch_arguments={
                'camera_name': name,
                'camera_namespace': 'realsense',
                'serial_no': serial,
                'enable_color': 'true',
                'enable_depth': 'true',
                'pointcloud.enable': pointcloud,
                'align_depth.enable': 'true',
                'enable_rgbd': rgbd,
                'enable_sync': 'true',
                "rgb_camera.color_profile":f"{width},{height},{fps}",
            }.items()
        )

    return LaunchDescription(declared_arguments + [
        make_camera('head', '_250122072330', width=424, height=240, fps=6),
        make_camera('left_hand', '_838212072778', width=320, height=240, fps=6),
        make_camera('right_hand', '_926522071700', width=320, height=240, fps=6),
    ])
