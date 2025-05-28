import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from RealsenseInterface import RealSenseCameraPublisher

if __name__ == "__main__":
    # Initialize the RealSense publisher
    ChannelFactoryInitialize(id = 0, networkInterface="enx00e04c6803cc")
    Head_publisher = RealSenseCameraPublisher(channel_name='realsense/Head', serial_number = "250122072330", InitChannelFactory=False)
    Larm_publisher = RealSenseCameraPublisher(channel_name='realsense/LArm', serial_number = "838212072778", InitChannelFactory=False)
    start_time = time.time()

    while True:
        Head_publisher.publish()
        Larm_publisher.publish()

        time.sleep(1/60)  # Publish at 60 Hz
        print(f"Publishing at {(time.time() - start_time):.2f}s", flush=True)