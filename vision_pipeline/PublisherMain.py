import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from RealsenseInterface import RealSenseCameraPublisher

if __name__ == "__main__":
    ChannelFactoryInitialize(networkInterface = "lo")
    # head_pub = RealSenseCameraPublisher(
    #     channel_name='realsense/Head',
    #     serial_number="250122072330",
    #     InitChannelFactory=False
    # )
    # larm_pub = RealSenseCameraPublisher(
    #     channel_name='realsense/LArm',
    #     serial_number="838212072778",
    #     InitChannelFactory=False
    # )
    pub = RealSenseCameraPublisher("realsense/camera", InitChannelFactory=False)
    period   = 1.0 / 60.0
    next_ts  = time.perf_counter()

    while True:
        # --- wait for our scheduled dispatch time ---
        now = time.perf_counter()
        sleep_for = next_ts - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            # we’re behind schedule; skip sleeping
            next_ts = now

        # --- do the work ---
        t0 = time.perf_counter()
        # head_pub.publish()
        # larm_pub.publish()
        pub.publish()
        t1 = time.perf_counter()

        # --- book-keep for next frame ---
        next_ts += period

        # --- diagnostics (optional) ---
        loop_time   = (t1 - t0) * 1000
        to_next_ms  = (next_ts - t1) * 1000
        print(f"publish {loop_time:.2f} ms, sleep {to_next_ms:.2f} ms → 60 Hz", flush=True)
