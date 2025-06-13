from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('vision_pipeline')
    vp_rviz_file = os.path.join(pkg_share, 'launch', 'vision_pipeline_rviz.launch.py')

    vision_pipeline_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(vp_rviz_file)
    )
    example_client_node = Node(
            package='vision_pipeline',
            executable='exampleclient',
            name='example_client_node',
            output='screen',
        )

    return LaunchDescription([example_client_node, vision_pipeline_rviz])
