import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2
import matplotlib.pyplot as plt
import os
import sys
import threading
ros_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(ros_dir, "..")
utils_dir = os.path.join(parent_dir, "utils")
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

from ros_utils import TFHandler
class MeasureNode(Node):
    def __init__(self, topics, types, qos, base_frame):
        super().__init__("FrequencyMeasureNode")
        self.tf_handler = TFHandler(self)
        self.data_dict = {}
        for topic, m_type, qos in zip(topics, types, qos):
            cb = lambda msg, captured_topic=topic: self.freq_callback(msg, captured_topic)
            self.create_subscription(m_type, topic, cb, qos)
            self.data_dict[topic] = []
            print(f"Subscribed to topic: {topic}")
        self.base_frame = base_frame

        # Private executor & spin thread for this node
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._spin_thread.start()
    def freq_callback(self, msg, topic):
        # print(f"recvived data from {topic}")
        t = self.get_clock().now().nanoseconds / 1e9 
        # print(f"{t=} {dir(t)=}")
        self.data_dict[topic].append(t)
        #print(f"{topic}:{dir(msg)=}")
        if hasattr(msg, 'header'):
            source_frame = msg.header.frame_id
            #tf = self.tf_handler.lookup_pose(self.base_frame, source_frame, msg.header.stamp)
            tf = self.tf_handler.lookup_pose(self.base_frame, source_frame, Time())
            
            if tf is not None and source_frame != self.base_frame:
                if source_frame in self.data_dict:
                    self.data_dict[source_frame].append(self.get_clock().now().nanoseconds / 1e9 )
                else:
                    self.data_dict[source_frame] = [self.get_clock().now().nanoseconds / 1e9 ]
            else:
                #print(f"tf lookup failed {source_frame}->{self.base_frame}")
                pass
    def get_data(self):
        return self.data_dict



def MeasureCameraFrequency(args=None):
    rclpy.init(args=args)
    base_frame = "map"
    topics = ["/realsense/head/color/image_raw/compressed", "/realsense/head/aligned_depth_to_color/image_raw/compressedDepth",
              "/realsense/head/depth/color/points", "/realsense/head/color/camera_info",
              "/realsense/left_hand/color/image_raw/compressed", "/realsense/left_hand/aligned_depth_to_color/image_raw/compressedDepth",
              "/realsense/left_hand/depth/color/points", "/realsense/left_hand/color/camera_info",
              "/realsense/accumulated_point_cloud"]
    types = [CompressedImage, CompressedImage, PointCloud2, CameraInfo, CompressedImage, CompressedImage, PointCloud2, CameraInfo, PointCloud2]
    sensor_data_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
    qos = [sensor_data_qos] * len(types)

    plt.ion()
    fig, ax = plt.subplots()
    
    try:
        node = MeasureNode(topics, types, qos, base_frame)
        


        while rclpy.ok():
            ax.clear()
            data = node.get_data()
            for topic, timestamps in data.items():
                if len(timestamps) > 0:
                    times = sorted(timestamps)
                    time_deltas = [timestamps[i] - timestamps[i-1] for i in range(1, len(times))]

                    ax.plot( times[1:], time_deltas,marker='.', linestyle='-', label=topic)
                    
            # --- Finalize Plot Appearance on each frame ---
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("dt (seconds)")
            ax.set_title("time vs time between messages")
            ax.legend(loc='lower left', fontsize='small')
            ax.set_ylim(0,1)
            ax.grid(True)


            plt.pause(1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
