cmake_minimum_required(VERSION 3.5)
project(vision_pipeline)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# This tells ament to invoke setup.py and install the Python package
ament_python_install_package(${PROJECT_NAME})

ament_package()
