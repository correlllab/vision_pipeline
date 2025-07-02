from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    rviz_config_path = os.path.join(
        get_package_share_directory('vision_pipeline'),
        'Rviz',
        'VP_RVIZ.rviz'
    )

    return LaunchDescription([
        ExecuteProcess(
            cmd=['rviz2', '-d', rviz_config_path],
            output='screen'
        ),
        Node(
            package='vision_pipeline',
            executable='vp',
            output='screen',
            emulate_tty=True
        )
    ])
