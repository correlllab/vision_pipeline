import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from RealsenseInterface import RealSenseCameraSubscriber

if __name__ == "__main__":
    # Initialize the RealSense Subscriber
    ChannelFactoryInitialize()
    subscriber = RealSenseCameraSubscriber(channel_name='realsense', InitChannelFactory=False)
    #Head_subscriber = RealSenseCameraSubscriber(channel_name='realsense/Head', InitChannelFactory=False)
    #Larm_subscriber = RealSenseCameraSubscriber(channel_name='realsense/LArm', InitChannelFactory=False)
    while True:
        #Head_subscriber.read(display=True)
        #Larm_subscriber.read(display=True)
        subscriber.read(display=True)
        time.sleep(1/60)
        print("Reading images at 60 Hz", flush=True)