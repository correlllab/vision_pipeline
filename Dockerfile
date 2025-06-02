# ~/ros2_ws/src/vision_pipeline/Dockerfile
FROM osrf/ros:humble-desktop-full

# 1) Make /ros2_ws the workspace root
WORKDIR /ros2_ws

# 2) (NO COPY HERE—we’ll mount src/ at runtime)

# 3) Build stub (this will be skipped on first build, but colcon is already installed)
RUN bash -lc "source /opt/ros/humble/setup.bash && \
             if [ -d /ros2_ws/src/vision_pipeline ]; then \
               colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release; \
             fi"

# 4) Create entrypoint that sources ROS and (later) any built workspace
RUN echo '#!/usr/bin/env bash'            > /ros_entrypoint.sh && \
    echo 'source /opt/ros/humble/setup.bash'  >> /ros_entrypoint.sh && \
    echo 'if [ -f /ros2_ws/install/setup.bash ]; then source /ros2_ws/install/setup.bash; fi' >> /ros_entrypoint.sh && \
    echo 'exec "$@"'                         >> /ros_entrypoint.sh && \
    chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
