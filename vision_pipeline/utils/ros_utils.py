import numpy as np
import cv2
import open3d as o3d
from geometry_msgs.msg import Point
from sensor_msgs.msg  import PointCloud2, PointField
from sensor_msgs_py   import point_cloud2
from std_msgs.msg import Header
import struct

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException
from tf2_ros         import LookupException, ConnectivityException, ExtrapolationException
from rclpy.time import Time
from rclpy.duration import Duration

from rclpy.node import Node
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R



def decode_compressed_depth_image(msg) -> np.ndarray:
    """
    Decodes a ROS2 compressed depth image (format: '16UC1; compressedDepth').

    Args:
        msg (CompressedImage): CompressedImage ROS message.

    Returns:
        np.ndarray: Decoded 16-bit depth image as a NumPy array.
    """
    # Ensure format is correct
    if not msg.format.lower().endswith("compresseddepth"):
        raise ValueError(f"Unsupported format: {msg.format}")

    # The first 12 bytes of the data are the depth image header
    header_size = 12
    if len(msg.data) <= header_size:
        raise ValueError("CompressedImage data too short to contain depth header")

    # Strip the custom header
    depth_header = msg.data[:header_size]
    compressed_data = msg.data[header_size:]

    # Optional: parse header (rows, cols, format, etc.) if needed
    # rows, cols, fmt, comp = struct.unpack('<II2B', depth_header[:10])

    # Decode the remaining PNG data into a 16-bit image
    np_arr = np.frombuffer(compressed_data, dtype=np.uint8)
    depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    if depth_image is None:
        raise ValueError("cv2.imdecode failed on compressed depth image")

    if depth_image.dtype != np.uint16:
        raise TypeError(f"Expected uint16 image, got {depth_image.dtype}")

    return depth_image

def box_to_points(box):
    # min and max
    min_pt = np.array(box.min_bound)
    max_pt = np.array(box.max_bound)

    # 8 corners
    corners = [
        [min_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]]
    ]

    # 12 edges as pairs of indices into corners list
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]

    # Build list of Points for LINE_LIST (pairs form lines)
    points = []
    for i, j in edges:
        p1 = Point(x=corners[i][0], y=corners[i][1], z=corners[i][2])
        p2 = Point(x=corners[j][0], y=corners[j][1], z=corners[j][2])
        points.append(p1)
        points.append(p2)
    return points

def _get_field_info(fields):
    field_info = {}
    for f in fields:
        field_info[f.name] = {
            'offset': f.offset,
            'datatype': f.datatype,
            'count': f.count,
            'size': _get_type_size(f.datatype)
        }
    return field_info

def _get_type_size(datatype):
    if datatype == PointField.INT8: return 1
    if datatype == PointField.UINT8: return 1
    if datatype == PointField.INT16: return 2
    if datatype == PointField.UINT16: return 2
    if datatype == PointField.INT32: return 4
    if datatype == PointField.UINT32: return 4
    if datatype == PointField.FLOAT32: return 4
    if datatype == PointField.FLOAT64: return 8
    return 0

def pcd_to_msg(pcd: o3d.t.geometry.PointCloud, frame_id: str) -> PointCloud2:
    """
    Converts an Open3D *Tensor* PointCloud (o3d.t.geometry.PointCloud) to a ROS2 PointCloud2.
    Assumes 'positions' exist; 'colors' optional (if absent, colors default to zeros).
    Encodes RGB as a packed float32 in a field named 'rgb'.
    """
    if "positions" not in pcd.point or pcd.point["positions"].shape[0] == 0:
        print("Warning: Tensor PointCloud has no positions to convert to ROS2 message.")
        return PointCloud2()

    # --- Pull tensor data as numpy ---
    pts = pcd.point["positions"].numpy()  # shape [N,3], dtype typically float32/float64
    if pts.dtype != np.float32:
        pts = pts.astype(np.float32, copy=False)

    N = pts.shape[0]

    if "colors" in pcd.point:
        cols = pcd.point["colors"].numpy()  # assume [0,1] float or [0..255] uint8
        if cols.dtype == np.uint8:
            rgb_uint8 = cols
        else:
            rgb_uint8 = np.clip(cols, 0.0, 1.0) * 255.0
            rgb_uint8 = rgb_uint8.astype(np.uint8)
    else:
        # Default to black if colors missing
        rgb_uint8 = np.zeros((N, 3), dtype=np.uint8)

    # Pack into 0xRRGGBB as uint32, then view as float32
    rgb_packed_uint32 = (
        (rgb_uint8[:, 0].astype(np.uint32) << 16) |
        (rgb_uint8[:, 1].astype(np.uint32) << 8)  |
        (rgb_uint8[:, 2].astype(np.uint32))
    )
    rgb_packed_float32 = rgb_packed_uint32.view(np.float32)

    # Build structured array for PointCloud2
    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('rgb', np.float32)
    ])

    pc_struct = np.zeros(N, dtype=dtype)
    pc_struct['x'] = pts[:, 0]
    pc_struct['y'] = pts[:, 1]
    pc_struct['z'] = pts[:, 2]
    pc_struct['rgb'] = rgb_packed_float32

    header = Header()
    header.frame_id = frame_id

    fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = N
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = pc_struct.dtype.itemsize
    msg.row_step = msg.point_step * N
    msg.is_dense = True
    msg.data = pc_struct.tobytes()
    return msg

def msg_to_pcd(msg: PointCloud2) -> o3d.t.geometry.PointCloud:
    """
    Converts a ROS2 PointCloud2 message (with fields x,y,z and packed float32 'rgb')
    into an Open3D *Tensor* PointCloud (o3d.t.geometry.PointCloud).
    """
    pcd_t = o3d.t.geometry.PointCloud()

    total_points = msg.width * msg.height
    if total_points == 0:
        print("Warning: Received empty PointCloud2 message.")
        return pcd_t

    field_info = _get_field_info(msg.fields)
    if not all(k in field_info for k in ("x", "y", "z")):
        print("Error: PointCloud2 missing 'x', 'y', or 'z' fields.")
        return pcd_t

    has_rgb = ('rgb' in field_info and
               field_info['rgb']['datatype'] == PointField.FLOAT32)

    endian_char = '<' if not msg.is_bigendian else '>'
    fmt_f32 = endian_char + 'f'

    xyz = np.empty((total_points, 3), dtype=np.float32)
    cols = np.empty((total_points, 3), dtype=np.float32) if has_rgb else None

    for i in range(total_points):
        base = i * msg.point_step
        x = struct.unpack_from(fmt_f32, msg.data, base + field_info['x']['offset'])[0]
        y = struct.unpack_from(fmt_f32, msg.data, base + field_info['y']['offset'])[0]
        z = struct.unpack_from(fmt_f32, msg.data, base + field_info['z']['offset'])[0]
        xyz[i, :] = (x, y, z)

        if has_rgb:
            rgb_f32 = struct.unpack_from(fmt_f32, msg.data, base + field_info['rgb']['offset'])[0]
            # Avoid 0-d dtype change issue by using a 1D array then viewing
            rgb_u32 = np.array([rgb_f32], dtype=np.float32).view(np.uint32)[0]
            r = (rgb_u32 >> 16) & 0xFF
            g = (rgb_u32 >> 8)  & 0xFF
            b = (rgb_u32)       & 0xFF
            cols[i, :] = (r/255.0, g/255.0, b/255.0)

    # Fill tensor point cloud
    pcd_t.point["positions"] = o3d.core.Tensor(xyz, dtype=o3d.core.Dtype.Float32)

    if has_rgb:
        if total_points > 0:
            pcd_t.point["colors"] = o3d.core.Tensor(cols, dtype=o3d.core.Dtype.Float32)
        else:
            print("Warning: 'rgb' field present but no color data extracted.")
    else:
        print("Info: 'rgb' field not present or not FLOAT32; only XYZ set.")

    return pcd_t

def transform_to_matrix(tf_msg):
    """
    Convert a geometry_msgs.msg.Transform into a 4×4 numpy array.
    """
    # Translation
    tx = tf_msg.translation.x
    ty = tf_msg.translation.y
    tz = tf_msg.translation.z

    # Quaternion
    qx = tf_msg.rotation.x
    qy = tf_msg.rotation.y
    qz = tf_msg.rotation.z
    qw = tf_msg.rotation.w

    # Build rotation matrix
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    R = np.array([
        [1 - 2*(yy + zz),   2*(xy - wz),      2*(xz + wy)],
        [2*(xy + wz),       1 - 2*(xx + zz),  2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),      1 - 2*(xx + yy)]
    ])

    # Assemble homogeneous matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T

class TFHandler:
    """
    A reusable class to handle TF2 transformations.
    """
    def __init__(self, node: Node, cache_time: float = 60.0):
        """
        Initializes the TF handler.
        
        Args:
            node: The ROS 2 node to attach the listener to.
            cache_time: How many seconds of past transforms to buffer.
        """
        self.node = node
        self._buffer = Buffer(cache_time=Duration(seconds=cache_time))
        # The TransformListener should be spun. Setting spin_thread=True handles this automatically.
        self._listener = TransformListener(self._buffer, node, spin_thread=True)
        # print(f"\n\n{dir(self._buffer)=}\n{dir(self._listener)=}")
    def lookup_transform(
        self,
        target_frame: str,
        source_frame: str,
        time_stamp: Time,
        timeout_sec: float = 0.1
    ):
        """
        Looks up the transform between two coordinate frames.

        Args:
            target_frame: The frame to transform into.
            source_frame: The frame to transform from.
            time_stamp: The time at which to get the transform. Use rclpy.time.Time() for latest.
            timeout_sec: How long to wait for the transform to become available.

        Returns:
            A geometry_msgs.msg.Transform object or None if the lookup fails.
        """
        # Normalize stamp to rclpy.time.Time if it's from a message header
        stamp = time_stamp if isinstance(time_stamp, Time) else Time.from_msg(time_stamp)
        #FUCKIT WE BALL
        # stamp = Time()  # latest available
        source_frame = "left_wrist_yaw_link"  # TEMP HACK
        try:
            
            # Perform the lookup
            xf = self._buffer.lookup_transform(target_frame, source_frame, stamp, timeout=Duration(seconds=timeout_sec))
            T = transform_to_matrix(xf.transform)
            return T

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.node.get_logger().info(
                f"TF lookup failed from '{source_frame}' to '{target_frame}': {e}",
                throttle_duration_sec=2.0
            )
            return None

    def lookup_pose(
        self,
        target_frame: str,
        source_frame: str,
        time_stamp: Time,
    ) -> list[float] | None:
        """
        Looks up the pose of a source frame relative to a target frame.

        Args:
            target_frame: The reference frame.
            source_frame: The frame whose pose is being requested.
            time_stamp: The time of the request. Use rclpy.time.Time() for the latest pose.

        Returns:
            A list [x, y, z, roll, pitch, yaw] in the target_frame, or None on failure.
        """
        tf = self.lookup_transform(target_frame, source_frame, time_stamp)
        if tf is None:
            return None

        t = tf[:3, 3]
        Rmat = tf[:3, :3]
        roll, pitch, yaw = R.from_matrix(Rmat).as_euler('xyz', degrees=False)

        return [t[0], t[1], t[2], roll, pitch, yaw]

def box_to_marker(box, color, frame, id):
    box_marker = Marker()
    box_marker.header.frame_id = frame   # or your preferred frame
    box_marker.ns = "boxes"
    box_marker.id = id
    box_marker.type = Marker.LINE_LIST
    box_marker.action = Marker.ADD
    box_marker.pose.orientation.w = 1.0  # No rotation
    box_marker.scale.x = 0.01            # Line width in meters
    box_marker.color.r = color[0]
    box_marker.color.g = color[1]
    box_marker.color.b = color[2]
    box_marker.color.a = color[3]
    box_marker.id = id
    #print(f"{dir(box)=}")
    #input("Press Enter to continue...")
    box_marker.points = box_to_points(box)
    return box_marker

def text_marker(text, position, color, frame, id):
    text_marker = Marker()
    text_marker.header.frame_id = frame
    text_marker.ns = "text"
    text_marker.id = id
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose.position.x = position[0]
    text_marker.pose.position.y = position[1]
    text_marker.pose.position.z = position[2]
    text_marker.pose.orientation.w = 1.0
    text_marker.scale.z = 0.05  # Font size
    text_marker.color.r = color[0]
    text_marker.color.g = color[1]
    text_marker.color.b = color[2]
    text_marker.color.a = color[3]
    text_marker.text = text
    return text_marker