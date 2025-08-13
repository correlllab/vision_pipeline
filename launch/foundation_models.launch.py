from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Find the package share directory
    pkg_share = get_package_share_directory("vision_pipeline")

    # Path to RViz config file
    rviz_config_path = os.path.join(pkg_share, "Rviz", "FoundationModels.rviz")

    return LaunchDescription([
        # Vision pipeline foundation_models node
        Node(
            package="vision_pipeline",
            executable="foundation_models",
            name="foundation_models_node",
            output="screen"
        ),

        # RViz2 with specified config
        ExecuteProcess(
            cmd=["rviz2", "-d", rviz_config_path],
            output="screen"
        )
    ])