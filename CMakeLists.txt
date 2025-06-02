cmake_minimum_required(VERSION 3.5)
project(vision_pipeline)

# Use ament_cmake_python for Python‐only packages
find_package(ament_cmake_python REQUIRED)

# Install everything under the "vision_pipeline" Python module
ament_python_install_package(${PROJECT_NAME})

# Install launch files so ros2 launch can find them
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
