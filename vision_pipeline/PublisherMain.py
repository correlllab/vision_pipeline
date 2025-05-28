import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RealsenseInterface import RealSenseCameraPublisher

if __name__ == "__main__":
    # Initialize the RealSense publisher
    ChannelFactoryInitialize()
    Head_publisher = RealSenseCameraPublisher(channel_name='realsense/Head', serial_number = "250122072330", InitChannelFactory=False)
    Larm_publisher = RealSenseCameraPublisher(channel_name='realsense/LArm', serial_number = "838212072778", InitChannelFactory=False)
    start_time = time.time()

    while True:
        Head_publisher.publish()
        Larm_publisher.publish()

        time.sleep(1/60)  # Publish at 60 Hz
        print(f"Publishing at {(time.time() - start_time):.2f}s", flush=True)