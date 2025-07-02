#!/usr/bin/env python3
import os
from setuptools import find_packages, setup
from glob import glob

package_name = 'vision_pipeline'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # include configuration file
        ('share/' + package_name, [os.path.join('vision_pipeline', 'config.json')]),
        ('share/' + package_name + '/launch', ['launch/realsense_cameras.launch.py', "launch/example_client.launch.py", "launch/vp.launch.py"]),
        ('share/' + package_name + '/Rviz', glob('Rviz/*.rviz')),

    ],
    install_requires=['setuptools', 'sensor_msgs', 'visualization_msgs', 'custom_ros_messages'],
    zip_safe=True,
    maintainer='todo',
    maintainer_email='todo@gmail.com',
    description='Vision Pipeline ROS2 package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera = vision_pipeline.RosRealsense:TestSubscriber',
            'foundationmodels = vision_pipeline.RosRealsense:TestFoundationModels',
            'visionpipeline = vision_pipeline.RosVisionPipeline:RunVisionPipe',
            'exampleclient = vision_pipeline.RosVisionPipeline:TestExampleClient',
            'main = vision_pipeline.main:main',
        ],
    },
)