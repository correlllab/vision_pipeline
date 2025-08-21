#!/bin/bash
sudo docker build --platform linux/arm64 -f "$@" -t vision-pipeline:latest .
