#!/bin/bash
xhost +local:root

docker run --rm -it\
  --network host \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  vision_pipeline:humble \
  "$@"


