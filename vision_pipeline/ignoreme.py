import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from RealsenseInterface import RealSenseCameraSubscriber

if __name__ == "__main__":
    # Initialize the RealSense publisher
    ChannelFactoryInitialize(id = 0, networkInterface="enx00e04c681314")

    print("Starting RealSense Camera Subscriber...")
    Head_subscriber = RealSenseCameraSubscriber(channel_name='realsense/Head', InitChannelFactory=False)
    print("Head subscriber initialized.")
    Larm_subscriber = RealSenseCameraSubscriber(channel_name='realsense/LArm', InitChannelFactory=False)
    print("Left arm subscriber initialized.")
    while True:
        Head_subscriber.read(display=True)
        Larm_subscriber.read(display=True)

        time.sleep(1/60)
        print("Reading images at 60 Hz", flush=True)