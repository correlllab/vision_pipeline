#!/bin/bash

docker run --rm -it \
  -v /home/unitree/vp_ws/src:/ros2_ws/src \
  --shm-size=1g \
  --network host \
  --cpus="4" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /etc/localtime:/etc/localtime:ro \
  -v /etc/timezone:/etc/timezone:ro \
  vision-pipeline:latest \
  "$@"
