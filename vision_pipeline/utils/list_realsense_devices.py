import pyrealsense2 as rs

# Create a context object to manage devices
context = rs.context()

# Get a list of all connected RealSense devices
devices = context.query_devices()

if len(devices) == 0:
    print("No RealSense devices found.")
else:
    for device in devices:
        print(f"Device: {device.get_info(rs.camera_info.name)}")
        print(f"Serial Number: {device.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware Version: {device.get_info(rs.camera_info.firmware_version)}")

        # Try to start a pipeline to get stream resolution
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device.get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.color, rs.format.bgr8, 30)  # Default FPS 30

        try:
            profile = pipeline.start(config)
            stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = stream.get_intrinsics()
            print(f"Resolution: {intrinsics.width}x{intrinsics.height}")
            pipeline.stop()
        except Exception as e:
            print(f"Could not retrieve resolution: {e}")

        print("-" * 30)
