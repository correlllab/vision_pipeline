import time
from vision_pipeline.RealsenseInterface import RealSenseCameraPublisher
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

if __name__ == "__main__":
    # Initialize the RealSense publisher
    ChannelFactoryInitialize()
    Head_publisher = RealSenseCameraPublisher(channel_name='realsense/head', serial_number = "" InitChannelFactory=False)
    Larm_publisher = RealSenseCameraPublisher(channel_name='realsense/head', serial_number = "" InitChannelFactory=False)

    while True:
        Head_publisher.publish()
        Larm_publisher.publish()
        time.sleep(1/60)  # Publish at 60 Hz
