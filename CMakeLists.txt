cmake_minimum_required(VERSION 3.5)
project(vision_pipeline)

# export Python modules via ament_python
find_package(ament_python REQUIRED)

# If you have any CMake/ROS-level includes, you’d put them here
# e.g. find_package(realsense2_camera REQUIRED) 
# but for a purely Python package you often don’t need to “find” in CMakeLists, because 
# you’re not compiling C++.

# Install Python modules from setup.py:
ament_python_install_package(${PROJECT_NAME})

# If you declare any launch files, you can export them here:
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Add tests or data folders as needed:
# install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)

ament_package()
