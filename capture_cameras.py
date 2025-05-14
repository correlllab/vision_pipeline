import cv2
import pyrealsense2 as rs
import numpy as np


class RealSenseCamera:
    """
    A cv2.VideoCapture–style wrapper around a RealSense pipeline that
    defaults to the camera’s native streams if no width/height/fps are given.

    Usage:
        # Use hardware defaults
        cam = RealSenseCamera()
        # Or specify resolution, framerate, and/or device serial
        cam = RealSenseCamera(width=640, height=480, fps=30, serial_number="1234567890")

        rgb = cam.read()                              # just color
        rgb, depth = cam.read(return_depth=True)      # color + depth
        intr = cam.get_intrinsics()                   # get camera intrinsics
        cam.release()
    """
    def __init__(self, width=None, height=None, fps=None,
                 serial_number: str = None, depth_scale: float = None):
        # Create and configure pipeline
        self.pipeline = rs.pipeline()
        cfg = rs.config()

        # If a specific device serial is provided, restrict to that camera
        if serial_number:
            cfg.enable_device(serial_number)

        # Configure color and depth streams (defaults or provided settings)
        if width and height and fps:
            cfg.enable_stream(rs.stream.color,  width, height, rs.format.bgr8, fps)
            cfg.enable_stream(rs.stream.depth,  width, height, rs.format.z16,  fps)
            self.profile = self.pipeline.start(cfg)
        else:
            # No args or incomplete args → use camera’s built-in defaults
            self.profile = self.pipeline.start()

        # Depth scale (override if provided)
        sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = sensor.get_depth_scale() if depth_scale is None else depth_scale

        # Align depth frame to color frame for pixel alignment
        self.align = rs.align(rs.stream.color)

    def read(self, return_depth: bool = False):
        """
        Grabs the next aligned color (and optional depth) frame.

        Args:
            return_depth: if True, also return a depth array (in meters)

        Returns:
            color_image [H×W×3 uint8] or (color_image, depth_image)
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            return (False, None, None) if return_depth else (False, None)
        
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        if return_depth:
            depth_frame = aligned.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth_image *= self.depth_scale
            return True, color_image, depth_image

        return True, color_image

    def get_intrinsics(self):
        """
        Returns the color camera intrinsics:
        {fx, fy, cx, cy, width, height, model, coeffs}
        """
        video_prof = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = video_prof.get_intrinsics()
        return {
            'fx': intr.fx,
            'fy': intr.fy,
            'cx': intr.ppx,
            'cy': intr.ppy,
            'width': intr.width,
            'height': intr.height,
            'model': intr.model,
            'coeffs': intr.coeffs
        }

    def release(self):
        """
        Stops the camera pipeline.
        """
        self.pipeline.stop()




def get_cap(index):
    """Open and configure a V4L2 capture at 3200×1200@60 fps MJPG."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    # Force MJPG at 3200×1200 @ 60 fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print(f"Error: Unable to access camera at index {index}.")
        return None

    # Verify settings
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Opened camera {index} at {w}x{h} @ {fps:.1f} fps")
    return cap

def display_cap(index):
    cap = get_cap(index)
    if cap is None:
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break

        cv2.imshow(f"Camera {index} ({frame.shape[1]},{frame.shape[0]})", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    #display_cap(0)
    # for i in range(10):
    #     display_cap(i)
    rs_camera = RealSenseCamera()
    while True:
        ret, color, depth = rs_camera.read(return_depth=True)
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break

        cv2.imshow("Color Frame", color)
        cv2.imshow("Depth Frame", depth)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
