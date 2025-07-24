#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import json

config_path = os.path.join(get_package_share_directory('vision_pipeline'), 'config.json')
config = json.load(open(config_path))


def make_camera(name: str, serial: str, width: int, height: int, fps: int) -> Node:
    """
    Factory function to create a RealSense camera node.
    All parameters are shown grouped by modality; most are commented out by default.
    """
    cam_idx = -1
    try:
        cam_idx = config["rs_names"].index(name)
    except ValueError:
        self.get_logger().error(f"Camera name {name} not found in config.")
        raise ValueError(f"Camera name {name} not found in config {config['rs_names']=}.")
    source_frame = config["rs_frames"][cam_idx]

    return Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name=name,
        namespace='realsense',
        output='screen',
        emulate_tty=True,
        parameters=[{
            # === Identification & Logging ===
            'camera_name': name,
            'serial_no': serial,
            'initial_reset': True,
            'base_frame_id':  f'{source_frame}',


            # === Modalities ===
            'enable_color':True,
            'enable_depth':True,
            'enable_sync':True,
            'align_depth.enable':True,
            'pointcloud.enable':True,
            'enable_accel':False,            
            'enable_gyro':False,
            'enable_infra1':False,
            'enable_infra2':False,
            'enable_rgbd':True,

            # === Plugins ===
            f'{name}.color.image_raw.enable_pub_plugins':      ['image_transport/compressed'],
            f'{name}.depth.image_rect_raw.enable_pub_plugins': ['image_transport/compressedDepth'],
            f'{name}.aligned_depth_to_color.image_raw.enable_pub_plugins': ['image_transport/compressedDepth'],


            # === Qos ===
            # 'accel_info_qos': 'SENSOR_DATA',
            # 'accel_qos': 'SENSOR_DATA',
            'color_info_qos': 'SENSOR_DATA',
            'color_qos': 'SENSOR_DATA',
            'depth_info_qos': 'SENSOR_DATA',
            'depth_qos': 'SENSOR_DATA',
            # 'gyro_info_qos': 'SENSOR_DATA',
            # 'gyro_qos': 'SENSOR_DATA',
            # 'infra1_info_qos': 'SENSOR_DATA',
            # 'infra1_qos': 'SENSOR_DATA',
            # 'infra2_info_qos': 'SENSOR_DATA',
            # 'infra2_qos': 'SENSOR_DATA',
            'pointcloud.pointcloud_qos': 'SENSOR_DATA',

            # === Profiles ===
            'depth_module.depth_profile': f"{width}x{height}x{fps}",
            # 'depth_module.infra_profile': f"{width}x{height}x{fps}",
            'rgb_camera.color_profile': f"{width}x{height}x{fps}",

            # === Filters ===
            'decimation_filter.enable': True,
            'decimation_filter.filter_magnitude': 2,
            'pointcloud.stream_filter':2,
            'pointcloud.stream_index_filter':0,
            # 'spatial_filter.enable': False,
            # 'temporal_filter.enable': False,
            # 'hole_filling_filter.enable': False,

            # === Transform & Playback ===
            'publish_tf': True,
            # 'tf_publish_rate': 50.0,
            # 'initial_reset': False,
            # 'json_file_path': '',
            # 'rosbag_filename': '',
            # 'rosbag_loop': False,
            # 'wait_for_device_timeout': -1.0,
            # 'reconnect_timeout': 6.0,
        }]
    )



def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()

    # Create one node per configured camera
    for cam_name, serial in zip(config['rs_names'], config['rs_serials']):
        node = make_camera(
            name=cam_name,
            serial=serial,
            width=config['rs_width'],
            height=config['rs_height'],
            fps=config['rs_fps'],
        )
        ld.add_action(node)

    pc_acc_node = Node(
        package='vision_pipeline',
        executable='pc_acc',
        name='pc_acc',
        output='screen',
        emulate_tty=True,

    )
    ld.add_action(pc_acc_node)

    return ld