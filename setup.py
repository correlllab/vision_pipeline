from setuptools import setup

package_name = 'vision_pipeline'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],  # installs the "vision_pipeline" folder
    data_files=[
        # Include package.xml so that ros2 can discover this as a ROS package
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # If you have config.json or other data:
        (f'share/{package_name}/config', ['vision_pipeline/config.json']),
    ],
    install_requires=[
        'torch>=2.0.0',           # your minimum Torch version
        'opencv-python>=4.5.0',   # ensure OpenCV-Python is installed
        # add any other pip packages here (e.g. numpy is usually pulled in by OpenCV or Torch)
    ],
    entry_points={
        'console_scripts': [
            # If you have Python nodes, e.g.:
            'robot_main = vision_pipeline.RobotMain:main',
            'realsense_interface = vision_pipeline.RealsenseInterface:main',
            # …add more node scripts as needed
        ],
    },
    zip_safe=True,
)
