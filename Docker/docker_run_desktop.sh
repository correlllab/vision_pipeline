#!/bin/bash
xhost +local:root

docker run --rm -it \
  -v /home/max/vp_ws/src:/ros2_ws/src \
  --gpus all \
  --shm-size=1g \
  --network host \
  --cpus="4" \
  --privileged \
  --device /dev/bus/usb:/dev/bus/usb \
  -v /dev:/dev \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /etc/localtime:/etc/localtime:ro \
  -v /etc/timezone:/etc/timezone:ro \
  vision-pipeline:latest \
  "$@"