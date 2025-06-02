#!/usr/bin/env python3
import numpy as np
import cv2

from dataclasses import dataclass
from cyclonedds.idl import IdlStruct
import pyrealsense2 as rs

import cyclonedds.idl.types as types
from typing import List
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize, ChannelSubscriber
import zlib

#print(f"{dir(types)=}")
@dataclass
class Image(IdlStruct, typename="Image"):
    """
    Represents an image with its raw data, width, height, and encoding.
    """
    width: types.uint32
    height: types.uint32
    data: types.sequence[types.uint8]#types.sequence[types.float32] # Using List[int] to represent byte data; DDS doesn't have a direct 'bytes' type, often uses sequence of octets (uint8)

@dataclass
class CameraSensorData(IdlStruct, typename="CameraSensorData"):
    """
    Combines RGB image, depth image, and camera matrices.
    """
    rgb_image: Image
    depth_image: Image
    intrinsic_matrix: types.sequence[types.float32] # A list of 9 floats for a 3x3 matrix
    extrinsic_matrix: types.sequence[types.float32] # A list of 16 floats for a 4x4 matrix

class RealSenseCameraPublisher:
    def __init__(self, channel_name, width=None, height=None, fps=60, serial_number: str = None, InitChannelFactory = True):

        if InitChannelFactory:
            ChannelFactoryInitialize()
        # Create and configure pipeline
        #print("here1")
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
            #print("here2")
            self.profile = self.pipeline.start(cfg)

        # Depth scale (override if provided)
        sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = sensor.get_depth_scale()

        # Align depth frame to color frame for pixel alignment
        self.align = rs.align(rs.stream.color)

        self.width = self.profile.get_stream(rs.stream.color).as_video_stream_profile().width()
        self.height = self.profile.get_stream(rs.stream.color).as_video_stream_profile().height()
        self.fps = self.profile.get_stream(rs.stream.color).as_video_stream_profile().fps()
        self.intrinsics = self.get_intrinsics()
        self.channel_name = channel_name
        self.publisher = ChannelPublisher(channel_name, CameraSensorData)
        self.publisher.Init()

    def publish(self, display=False):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            color_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = aligned.get_depth_frame()

        if not depth_frame:
            depth_frame = np.zeros((self.height, self.width), dtype=np.uint16)

        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_image *= self.depth_scale
        #print(f"depth: {type(depth_image)} {depth_image.shape} {depth_image.dtype} {depth_image.min()} {depth_image.max()}")
        #print(f"color {type(color_image)} {color_image.shape} {color_image.dtype} {color_image.min()} {color_image.max()}")
        #w = 100
        #h = 100
        #color_image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        #depth_image = (np.random.rand(h, w) * 255).astype(np.float32)
        if display:
            rgb_img_display = cv2.cvtColor(color_image.copy(), cv2.COLOR_BGR2RGB)
            cv2.imshow(f"Pub {self.channel_name}_RGB Image", rgb_img_display)
            cv2.imshow(f"Pub {self.channel_name}_Depth Image", depth_image)
            cv2.waitKey(1)
        color_bytes = color_image.tobytes()
        depth_bytes = depth_image.tobytes()

        compressed_color = zlib.compress(color_bytes)
        compressed_depth = zlib.compress(depth_bytes)
        #print(f"color_bytes: {len(color_bytes)} depth_bytes: {len(depth_bytes)}")
        #print(f"{type(color_bytes)} {type(depth_bytes)}")

        msg = CameraSensorData(
            rgb_image=Image(
                width=self.width,
                height=self.height,
                data=compressed_color#color_image.flatten().tolist()  # Flatten to 1D list
            ),
            depth_image=Image(
                width=self.width,
                height=self.height,
                data=compressed_depth#depth_image.flatten().tolist()  # Flatten to 1D list
            ),
            intrinsic_matrix=self.get_intrinsics(),
            extrinsic_matrix=self.get_extrinsics()
        )
        #print(f"Publishing message: {type(msg)}")
        self.publisher.Write(msg)
    def get_intrinsics(self):
        """
        Returns the color camera intrinsics:
        {fx, fy, cx, cy, width, height, model, coeffs}
        """
        video_prof = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = video_prof.get_intrinsics()
        """return {
            'fx': intr.fx,
            'fy': intr.fy,
            'cx': intr.ppx,
            'cy': intr.ppy,
            'width': intr.width,
            'height': intr.height,
            'model': intr.model,
            'coeffs': intr.coeffs
        }"""
        return [intr.fx, 0, intr.ppx,
                0, intr.fy, intr.ppy,
                0, 0, 1]
    def get_extrinsics(self):
        """
        Returns the extrinsic matrix from color to depth camera.
        The extrinsics are returned as a flattened 1D list.
        """
        extr = np.eye(4, dtype=np.float32)  # Default to identity
        return extr.flatten().tolist()  # Flatten to 1D list

    def release(self):
        """
        Stops the camera pipeline.
        """
        self.pipeline.stop()


class RealSenseCameraSubscriber():
    """
    Subscribes to the RGBD and CameraInfo topics for a given camera namespace under /realsense.
    Stores the latest RGB-D images and camera info for display or further processing.

    camera_key: e.g. 'head' or 'left_hand'.
    """
    def __init__(self, channel_name, InitChannelFactory=True):
        if InitChannelFactory:
            ChannelFactoryInitialize()
        self.channel_name = channel_name
        self.subscriber = ChannelSubscriber(channel_name, CameraSensorData)
        self.subscriber.Init()

    def read(self, display=False):
        msg = self.subscriber.Read(0.5)  # 0.5 seconds timeout
        if msg is None:
            print(f"No message received on channel {self.channel_name}.")
            return None, None, None, None
        rgb_bytes = zlib.decompress(bytes(msg.rgb_image.data))
        rgb_image = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(msg.rgb_image.height, msg.rgb_image.width, 3)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)


        depth_bytes = zlib.decompress(bytes(msg.depth_image.data))
        depth_image = np.frombuffer(depth_bytes, dtype=np.float32).reshape(msg.depth_image.height, msg.depth_image.width)


        #rgb_image = np.array(msg.rgb_image.data, dtype=np.uint8).reshape((msg.rgb_image.height, msg.rgb_image.width, 3))
        #depth_image = np.array(msg.depth_image.data, dtype=np.float32).reshape((msg.depth_image.height, msg.depth_image.width))
        #rgb_image = np.zeros((msg.rgb_image.height, msg.rgb_image.width, 3), dtype=np.uint8)
        #depth_image = np.zeros((msg.depth_image.height, msg.depth_image.width), dtype=np.float32)
        Intrinsics = np.array(msg.intrinsic_matrix, dtype=np.float32).reshape((3, 3))
        Extrinsics = np.array(msg.extrinsic_matrix, dtype=np.float32).reshape((4, 4))
        if display:
            print(f"Received message: {type(msg)}")
            cv2.imshow(f"Sub {self.channel_name}_RGB Image", rgb_image)
            cv2.imshow(f"Sub {self.channel_name}_Depth Image", depth_image)
            cv2.waitKey(1)
        return rgb_image, depth_image, Intrinsics, Extrinsics
    def shutdown(self):
        self.subscriber.Close()

if __name__ == "__main__":
    ChannelFactoryInitialize()
    pub = RealSenseCameraPublisher("realsense/camera", serial_number=None, InitChannelFactory=False)
    sub = RealSenseCameraSubscriber("realsense/camera", InitChannelFactory=False)
    while True:
        pub.publish(display=True)
        color, depth, intrinsics, extrinsics = sub.read(display=True)
