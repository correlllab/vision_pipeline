#!/bin/bash

cd ~/VisionPipeline/vision_pipeline/Publishers

# Function to handle SIGINT (Ctrl+C)
cleanup() {
    echo "Stopping background processes..."
    kill $LARM_PID $HEAD_PID
    wait
    exit 0
}

# Trap SIGINT and call cleanup
trap cleanup SIGINT

# Run publishers in background
python3 LArmPublisher.py &
LARM_PID=$!

python3 HeadPublisher.py &
HEAD_PID=$!

# Wait for both processes
wait
