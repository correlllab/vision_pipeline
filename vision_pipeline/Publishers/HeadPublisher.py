import time
from RealsenseInterface import RealSenseCameraPublisher
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

if __name__ == "__main__":
    # Initialize the RealSense publisher
    ChannelFactoryInitialize()
    Head_publisher = RealSenseCameraPublisher(channel_name='realsense/Head', serial_number = "250122072330", InitChannelFactory=False)
    start_time = time.time()
    while True:
        Head_publisher.publish()
        time.sleep(1/60)  # Publish at 60 Hz
        print(f"Head Publishing at {(time.time() - start_time):.2f}s")