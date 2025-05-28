import time
from RealsenseInterface import RealSenseCameraPublisher
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

if __name__ == "__main__":
    # Initialize the RealSense publisher
    ChannelFactoryInitialize()
    Larm_publisher = RealSenseCameraPublisher(channel_name='realsense/LArm', serial_number = "838212072778", InitChannelFactory=False)
    start_time = time.time()
    while True:
        Larm_publisher.publish()
        time.sleep(1/60)  # Publish at 60 Hz
        print(f"Larm Publishing at {(time.time() - start_time):.2f}s")