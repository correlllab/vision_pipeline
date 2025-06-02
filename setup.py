from setuptools import setup

package_name = 'vision_pipeline'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=[
        # Non-ROS PyPI dependencies (if any), e.g.:
        # 'numpy', 'opencv-python', 'torch'
    ],
    zip_safe=True,
    author='Max Conway',
    author_email='max@your_email.com',
    description='ROS2 Python vision pipeline nodes + launch files + RosWrappers',
    license='Apache-2.0',
    tests_require=['pytest'],

    # ──────────────────────────────────────────────────────────────────────────
    # 1) Entry points for any console scripts you want from your .py files
    # ──────────────────────────────────────────────────────────────────────────
    entry_points={
        'console_scripts': [
            # Existing nodes:
            'publisher_main       = vision_pipeline.PublisherMain:main',
            'robot_main           = vision_pipeline.RobotMain:main',
            'vision_pipeline_node = vision_pipeline.VisionPipeline:main',

            # Expose RosWrappers functions. 
            # Replace `main_wrapper` below with the actual function name(s) you have.
            # e.g. if RosWrappers.py has def main_wrapper(args=None): …, then:
            'ros_wrappers_main    = vision_pipeline.RosWrappers:main_wrapper',
            # If you have additional functions in RosWrappers, add them similarly:
            # 'another_wrapper   = vision_pipeline.RosWrappers:another_function',
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # 2) Data files: make sure the launch/ folder is installed under share/vision_pipeline/launch
    # ──────────────────────────────────────────────────────────────────────────
    data_files=[
        # Install any launch files so that "ros2 launch vision_pipeline <file>" works:
        ('share/vision_pipeline/launch', [
            'launch/realsense_cameras.launch.py',
        ]),
        # If you have other non-Python files (e.g. params/, configs/) you can install them here:
        # ('share/vision_pipeline/config', ['vision_pipeline/config.json']),
    ],
)
