#!/bin/bash
xhost +local:root

docker run --rm -it \
  --gpus all \
  --shm-size=1g \
  --network host \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /etc/localtime:/etc/localtime:ro \
  -v /etc/timezone:/etc/timezone:ro \
  vision_pipeline:humble \
  "$@"


