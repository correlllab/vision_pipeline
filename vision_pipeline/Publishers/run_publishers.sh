#!/bin/bash

cd ~/VisionPipeline/vision_pipeline/Publishers

# Function to handle Ctrl+C
cleanup() {
    echo "Stopping background processes..."
    kill -- -$LARM_PGID
    kill -- -$HEAD_PGID
    wait
    exit 0
}

# Trap SIGINT (Ctrl+C) to cleanup child process groups
trap cleanup SIGINT

# Start processes in their own process groups using setsid
setsid python3 LArmPublisher.py &
LARM_PID=$!
LARM_PGID=$(ps -o pgid= $LARM_PID | grep -o '[0-9]*')

setsid python3 HeadPublisher.py &
HEAD_PID=$!
HEAD_PGID=$(ps -o pgid= $HEAD_PID | grep -o '[0-9]*')

# Wait for both
wait
