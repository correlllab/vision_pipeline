# ~/ros2_ws/src/vision_pipeline/Dockerfile
FROM osrf/ros:humble-desktop-full
WORKDIR /ros2_ws

# 0) Remove any old/expired ROS 2 apt entry
RUN rm /etc/apt/sources.list.d/ros2-latest.list || true

# 1) Install curl/gnupg/lsb-release (for ROS 2 key) + python3-pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl gnupg2 lsb-release \
      python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 2) Re-import the ROS 2 GPG key and re-add the Jammy repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
    | gpg --dearmor \
    | tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu \
      $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/ros2-latest.list


 RUN apt-get update && \
      apt-get install -y --no-install-recommends \
      ros-humble-rmw-cyclonedds-cpp && \
      rm -rf /var/lib/apt/lists/*

# 3) Install Librealsense2 + ROS 2 realsense packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-humble-librealsense2* \
      ros-humble-realsense2-camera-msgs \
      ros-humble-realsense2-camera && \
    rm -rf /var/lib/apt/lists/*

# 4) Hard-code Python packages via pip
RUN pip3 install --no-cache-dir \
      "numpy<2" \
      opencv-python \
      open3d \
      torch \
      torchvision \
      torchaudio \
      matplotlib \
      transformers 



# 5) Copy your vision_pipeline source into the image
COPY src/vision_pipeline /ros2_ws/src/vision_pipeline

# 6) Build the workspace (ROS 2 + Python deps will be picked up)
RUN bash -lc "source /opt/ros/humble/setup.bash && \
             if [ -d /ros2_ws/src/vision_pipeline ]; then \
               colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release; \
             fi"

RUN git clone https://github.com/facebookresearch/sam2.git /ros2_ws/sam2 && \
    pip3 install --no-cache-dir -e /ros2_ws/sam2

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI="<CycloneDDS><Domain><General><!-- only use your Wi-Fi card --><Interfaces><NetworkInterface name=\"wlp5s0\" priority=\"default\" multicast=\"default\"/></Interfaces></General></Domain></CycloneDDS>"

# 7) Create an entrypoint that sources ROS 2 and your overlay
RUN echo '#!/usr/bin/env bash'            > /ros_entrypoint.sh && \
    echo 'source /opt/ros/humble/setup.bash'  >> /ros_entrypoint.sh && \
    echo 'if [ -f /ros2_ws/install/setup.bash ]; then source /ros2_ws/install/setup.bash; fi' >> /ros_entrypoint.sh && \
    echo 'exec "$@"'                         >> /ros_entrypoint.sh && \
    chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
