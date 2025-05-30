import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from RealsenseInterface import RealSenseCameraSubscriber

if __name__ == "__main__":
    ChannelFactoryInitialize(networkInterface= "wlp5s0")
    Head_sub = RealSenseCameraSubscriber(
        channel_name="realsense/Head",
        InitChannelFactory=False
    )
    LArm_sub = RealSenseCameraSubscriber(
        channel_name="realsense/LArm",
        InitChannelFactory=False
    )
    # sub = RealSenseCameraSubscriber("realsense/camera", InitChannelFactory=False)
    period  = 1.0 / 60.0
    next_ts = time.perf_counter()

    while True:
        # wait until it’s time for the next read
        now = time.perf_counter()
        to_sleep = next_ts - now
        if to_sleep > 0:
            time.sleep(to_sleep)
        else:
            # behind schedule, drop the delay
            next_ts = now

        # perform the read/display
        t0 = time.perf_counter()
        Head_sub.read(display=True)
        LArm_sub.read(display=True)
        # sub.read(display=True)
        t1 = time.perf_counter()

        # schedule the next iteration
        next_ts += period

        # optional debug: show how long read took and how much sleep remains
        work_ms   = (t1 - t0) * 1000
        sleep_ms  = (next_ts - t1) * 1000
        print(f"read {work_ms:.2f} ms, sleep {sleep_ms:.2f} ms → 60 Hz", flush=True)
