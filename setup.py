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
        ('share/' + package_name + '/launch', ["launch/vp.launch.py", #launches vision pipeline + VP.rviz
                                                "launch/foundation_models.launch.py" #launches foundation_models + foundation_models.rviz
                                                ]), 
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
            'camera = vision_pipeline.ROS.RosRealsense:TestSubscriber', #tests cameras in cv2
            'calibrate_camera = vision_pipeline.experiments.calibrate_camera:main', #moves the arms with vision
            'hz = vision_pipeline.ROS.frequency_measure:MeasureCameraFrequency', #tests subscription timing
            'foundation_models = vision_pipeline.ROS.RosRealsense:TestFoundationModels', #publishes topic to be used with FoundationModels.rviz
            'vp = vision_pipeline.ROS.RosVisionPipeline:RunVisionPipe', #executes main vision pipeline
            'exampleclient = vision_pipeline.ROS.RosVisionPipeline:TestExampleClient', #terminal based interactionwith vision pipeline
            'openbox = vision_pipeline.experiments.OpenBoxHardCode:main', #moves the arms with vision
            'prop_exp = vision_pipeline.experiments.proposed:main', #moves the arms with vision
        ],
    },
)