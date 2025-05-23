#!/usr/bin/env python3
import os
from setuptools import find_packages, setup

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='max',
    maintainer_email='maxlconway@gmail.com',
    description='Vision Pipeline ROS2 package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'capture_cameras = vision_pipeline.capture_cameras:main',
            'foundation_models = vision_pipeline.foundation_models:main',
            'subscriber = vision_pipeline.subscriber:main',
            'visionpipeline = vision_pipeline.visionpipeline:main',
            'FoundationModels_RosWrapper = vision_pipeline.FoundationModels_RosWrapper:main',
            'vision_pipeline_ros_wrapper = vision_pipeline.VisionPipeline_RosWrapper:main',
        ],
    },
)

