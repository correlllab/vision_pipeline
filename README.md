# VisionPipeline
This is a work in progress repo that holds a probabilistic vision pipeline
## Starting the camera topics
to start the camera topics ssh into your robot where the realsense are connected and create a vision pipleine workspace
```
user@robot mkdir ~/vp_ws/
user@robot mkdir ~/vp_ws/src
user@robot cd ~/vp_ws/src/
```
then clone and build the vision pipeline and custom messages
```
user@robot git clone git@github.com:correlllab/vision_pipeline.git
user@robot git clone git@github.com:correlllab/custom_ros_messages.git
user@robot cd ~/vp_ws/
user@robot colcon build
user@robot source ./install/setup.sh
user@robot ros2 launch vision_pipeline realsense_cameras.launch.py launch_head:=true launch_left_hand:=true  launch_right_hand:=true
```
you should see all the topics start, you can verify this in a new terminal with 
```
user@robot ros2 topic list
...
/realsense/head/aligned_depth_to_color/camera_info
/realsense/head/aligned_depth_to_color/image_raw
/realsense/head/aligned_depth_to_color/image_raw/compressed
/realsense/head/aligned_depth_to_color/image_raw/compressedDepth
/realsense/head/aligned_depth_to_color/image_raw/theora
/realsense/head/color/camera_info
/realsense/head/color/image_raw
/realsense/head/color/image_raw/compressed
/realsense/head/color/image_raw/compressedDepth
/realsense/head/color/image_raw/theora
/realsense/head/color/metadata
/realsense/head/depth/camera_info
/realsense/head/depth/image_rect_raw
/realsense/head/depth/image_rect_raw/compressed
/realsense/head/depth/image_rect_raw/compressedDepth
/realsense/head/depth/image_rect_raw/theora
/realsense/head/depth/metadata
/realsense/head/extrinsics/depth_to_color
/realsense/head/extrinsics/depth_to_depth
/realsense/left_hand/aligned_depth_to_color/camera_info
/realsense/left_hand/aligned_depth_to_color/image_raw
/realsense/left_hand/aligned_depth_to_color/image_raw/compressed
/realsense/left_hand/aligned_depth_to_color/image_raw/compressedDepth
/realsense/left_hand/aligned_depth_to_color/image_raw/theora
/realsense/left_hand/color/camera_info
/realsense/left_hand/color/image_raw
/realsense/left_hand/color/image_raw/compressed
/realsense/left_hand/color/image_raw/compressedDepth
/realsense/left_hand/color/image_raw/theora
/realsense/left_hand/color/metadata
/realsense/left_hand/depth/camera_info
/realsense/left_hand/depth/image_rect_raw
/realsense/left_hand/depth/image_rect_raw/compressed
/realsense/left_hand/depth/image_rect_raw/compressedDepth
/realsense/left_hand/depth/image_rect_raw/theora
/realsense/left_hand/depth/metadata
/realsense/left_hand/extrinsics/depth_to_color
/realsense/right_hand/aligned_depth_to_color/camera_info
/realsense/right_hand/aligned_depth_to_color/image_raw
/realsense/right_hand/aligned_depth_to_color/image_raw/compressed
/realsense/right_hand/aligned_depth_to_color/image_raw/compressedDepth
/realsense/right_hand/aligned_depth_to_color/image_raw/theora
/realsense/right_hand/color/camera_info
/realsense/right_hand/color/image_raw
/realsense/right_hand/color/image_raw/compressed
/realsense/right_hand/color/image_raw/compressedDepth
/realsense/right_hand/color/image_raw/theora
/realsense/right_hand/color/metadata
/realsense/right_hand/depth/camera_info
/realsense/right_hand/depth/image_rect_raw
/realsense/right_hand/depth/image_rect_raw/compressed
/realsense/right_hand/depth/image_rect_raw/compressedDepth
/realsense/right_hand/depth/image_rect_raw/theora
/realsense/right_hand/depth/metadata
/realsense/right_hand/extrinsics/depth_to_color
...

```
## Getting started
First create a vision pipeline workspace with
```
user@desktop mkdir ~/vp_ws/
user@desktop mkdir ~/vp_ws/src
```
then clone the vision pipeline and the custom ros messages into the src
```
user@desktop cd ~/vp_ws/src/
user@desktop git clone git@github.com:correlllab/vision_pipeline.git
user@desktop git clone git@github.com:correlllab/custom_ros_messages.git
```

Next edit the docker container with
```
user@desktop nano ~/vp_ws/src/vision_pipeline/Docker/Dockerfile
```
change the line
```
ENV CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"enp4s0\" priority=\"default\" multicast=\"default\"/></Interfaces></General></Domain></CycloneDDS>"
```
to use `name=name of your network interface`

finally build the docker container with
```
user@desktop cd ~/vp_ws/
user@desktop ./src/vision_pipeline/Docker/docker_build.sh ./src/vision_pipeline/Docker/DockerFile
```
To resolve an error like
```
> [ 6/15] RUN apt-get update &&     apt-get install -y --no-install-recommends       ros-humble-rmw-cyclonedds-cpp &&     rm -rf /var/lib/apt/lists/*:        
0.131 E: Conflicting values set for option Signed-By regarding source http://packages.ros.org/ros2/ubuntu/ jammy: /usr/share/keyrings/ros-archive-keyring.gpg !=
-----BEGIN PGP PUBLIC KEY BLOCK-----    
...
0.131    -----END PGP PUBLIC KEY BLOCK-----
0.131 E: The list of sources could not be read.
------
Dockerfile:27
--------------------
  26 |     # Install Cyclone DDS
  27 | >>> RUN apt-get update && \
  28 | >>>     apt-get install -y --no-install-recommends \
  29 | >>>       ros-humble-rmw-cyclonedds-cpp && \
  30 | >>>     rm -rf /var/lib/apt/lists/*
  31 |    
--------------------
ERROR: failed to solve: process "/bin/sh -c apt-get update &&     apt-get install -y --no-install-recommends       ros-humble-rmw-cyclonedds-cpp &&     rm -rf /var/lib/apt/lists/*" did not complete successfully: exit code: 100

```
comment out the lines in the DockerFile
```
# Re-import ROS 2 key and repo
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
    | gpg --dearmor \
    | tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu \
      $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/ros2-latest.list
```

to resolve an error like
```
Package ros-humble-rmw-cyclonedds-cpp is not available, but is referred to by another package.
This may mean that the package is missing, has been obsoleted, or
is only available from another source

E: Package 'ros-humble-rmw-cyclonedds-cpp' has no installation candidate
The command '/bin/sh -c apt-get update &&     apt-get install -y --no-install-recommends       ros-humble-rmw-cyclonedds-cpp &&     rm -rf /var/lib/apt/lists/*' returned a non-zero code: 100
```
make sure the lines in the DockerFile
```
# Re-import ROS 2 key and repo
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
    | gpg --dearmor \
    | tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu \
      $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/ros2-latest.list
```
are not commented out.

With the DockerFile build we just need to run the container
modify the file `vision_pipeline/Docker/docker_run.sh` so that the line ` -v /home/max/vp_ws/src/vision_pipeline:/ros2_ws/src/vision_pipeline` mounts your vision_pipeline, you likely should only have to change the `/home/max/` part

finally you can run the interactive container with
```
user@desktop ./src/vision_pipeline/Docker/docker_run.sh
```
you should see colcon build the vision pipeline and custom message packages like
```
user@desktop:~/vp_ws$ ./src/vision_pipeline/Docker/docker_run.sh 
non-network local connections being added to access control list
Sourcing ROS 2...
Building vision_pipeline...
Starting >>> custom_ros_messages
Starting >>> vision_pipeline
Finished <<< vision_pipeline [0.46s]                                    
Finished <<< custom_ros_messages [4.31s]                    

Summary: 2 packages finished [4.41s]
Sourcing overlay workspace...
root@DockerContainer:/ros2_ws# 
```

from there you can run vision_pipeline entry points with
```
root@DockerContainer:/ros2_ws# ros2 run vision_pipeline camera
root@DockerContainer:/ros2_ws# ros2 run vision_pipeline foundationmodels
root@DockerContainer:/ros2_ws# ros2 run vision_pipeline visionpipeline
```

to start the visionpipeline you can just use
```
user@desktop ./src/vision_pipeline/Docker/docker_run.sh ros2 run vision_pipeline visionpipeline 
```
## Code Explanaition

## Usage
