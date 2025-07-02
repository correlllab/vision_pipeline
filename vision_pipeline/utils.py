import torch
import numpy as np
import cv2
import open3d as o3d
from geometry_msgs.msg import Point
from sensor_msgs.msg  import PointCloud2, PointField
from sensor_msgs_py   import point_cloud2
from std_msgs.msg import Header
import struct



def get_points_and_colors(depths, rgbs, fx, fy, cx, cy):
    """
    Back-project a batch of depth and RGB images to 3D point clouds.

    Args:
        depths: Tensor of shape (B, H, W) representing depth in meters.
        rgbs: Tensor of shape (B, H, W, 3) representing RGB colors, range [0, 1] or [0, 255].
        fx, fy, cx, cy: camera intrinsics.

    Returns:
        points: Tensor of shape (B, H*W, 3) representing 3D points.
        colors: Tensor of shape (B, H*W, 3) representing RGB colors for each point.
    """
    B, H, W = depths.shape
    device = depths.device

    # Create meshgrid of pixel coordinates
    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')  # shape (H, W)

    # Flatten pixel coordinates
    grid_u_flat = grid_u.reshape(-1)  # (H*W,)
    grid_v_flat = grid_v.reshape(-1)  # (H*W,)

    # Flatten depth and color
    z = depths.reshape(B, -1)  # (B, H*W)
    colors = rgbs.reshape(B, -1, 3)  # (B, H*W, 3)

    # Back-project to camera coordinates
    x = (grid_u_flat[None, :] - cx) * z / fx  # (B, H*W)
    y = (grid_v_flat[None, :] - cy) * z / fy  # (B, H*W)

    # Stack into point sets
    points = torch.stack((x, y, z), dim=-1)  # (B, H*W, 3)

    return points, colors


def iou_2d(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute the 2D Intersection over Union (IoU) of two axis-aligned boxes.

    Args:
        box1: array_like of shape (4,), [xmin, ymin, xmax, ymax]
        box2: array_like of shape (4,), [xmin, ymin, xmax, ymax]

    Returns:
        IoU value (float) in [0.0, 1.0].
    """
    # Ensure inputs are numpy arrays
    b1 = np.array(box1, dtype=np.float64)
    b2 = np.array(box2, dtype=np.float64)

    # Intersection rectangle
    inter_xmin = max(b1[0], b2[0])
    inter_ymin = max(b1[1], b2[1])
    inter_xmax = min(b1[2], b2[2])
    inter_ymax = min(b1[3], b2[3])

    # Compute intersection width and height (clamp to zero if no overlap)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h  # area of overlap :contentReference[oaicite:0]{index=0}

    # Areas of the input boxes
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])

    # Union area
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0  # avoid division by zero :contentReference[oaicite:1]{index=1}

    # IoU is overlap divided by union :contentReference[oaicite:2]{index=2}
    return inter_area / union_area

def iou_3d(bbox1: o3d.geometry.AxisAlignedBoundingBox, bbox2: o3d.geometry.AxisAlignedBoundingBox) -> float:
    """
    Compute the 3D Intersection over Union (IoU) of two Open3D axis-aligned bounding boxes.

    Args:
        bbox1: An open3d.geometry.AxisAlignedBoundingBox instance.
        bbox2: An open3d.geometry.AxisAlignedBoundingBox instance.

    Returns:
        IoU value as a float in [0.0, 1.0].
    """
    # Get the min and max corner coordinates of each box
    min1 = np.array(bbox1.get_min_bound(), dtype=np.float64)
    max1 = np.array(bbox1.get_max_bound(), dtype=np.float64)
    min2 = np.array(bbox2.get_min_bound(), dtype=np.float64)
    max2 = np.array(bbox2.get_max_bound(), dtype=np.float64)

    # Compute the intersection box bounds
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    # Compute intersection dimensions (clamp to zero if no overlap)
    inter_dims = np.clip(inter_max - inter_min, a_min=0.0, a_max=None)
    inter_vol = np.prod(inter_dims)

    # Compute volumes of each box
    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)

    # Compute union volume
    union_vol = vol1 + vol2 - inter_vol
    if union_vol <= 0:
        return 0.0

    return float(inter_vol / union_vol)


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """
    Convert a 6-vector [x, y, z, roll, pitch, yaw] (radians)
    into a 4×4 homogeneous transform.
    """
    pose = [float(p) for p in pose]  # ensure float type
    x, y, z, roll, pitch, yaw = pose

    # Rotation about X axis (roll)
    Rx = np.array([
        [1,            0,             0],
        [0,  np.cos(roll), -np.sin(roll)],
        [0,  np.sin(roll),  np.cos(roll)],
    ])

    # Rotation about Y axis (pitch)
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])

    # Rotation about Z axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1],
    ])

    # Combined rotation: R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """
    Convert a 4×4 homogeneous matrix back into a 6-vector
    [x, y, z, roll, pitch, yaw] with angles in radians.
    Assumes T[:3,:3] = Rz @ Ry @ Rx.
    """
    # translation
    x, y, z = T[:3, 3]

    # rotation matrix
    R = T[:3, :3]

    # recover pitch = asin(–R[2,0])
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))

    # to avoid gimbal‐lock edge cases you could test cos(pitch)≈0
    # but for most cases:
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw  = np.arctan2(R[1, 0], R[0, 0])

    return np.array([x, y, z, roll, pitch, yaw])

def quat_to_euler(x, y, z, w):
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return np.array([roll, pitch, yaw])

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

def in_image(point: np.ndarray,
             obs_pose: np.ndarray,
             I: dict) -> bool:
    """
    Check if a 3D point in world coordinates projects into the image frame.

    Args:
        point: (3,) array giving the 3D point in world coordinates.
        obs_pose: (6,) array or list [x, y, z, roll, pitch, yaw] for camera pose in world.
        I: dict with camera intrinsics:
            - 'fx', 'fy': focal lengths
            - 'cx', 'cy': principal point
            - 'width', 'height': image size
            - 'model', 'coeffs': distortion model & parameters (ignored here if coeffs==0)

    Returns:
        True if the point projects within [0,width)×[0,height) and z_cam>0, else False.
    """
    # 1) build the camera-to-world transform, then invert to get world→camera
    T_cam2world = pose_to_matrix(obs_pose)
    T_world2cam = np.linalg.inv(T_cam2world)

    # 2) homogeneous world point → camera frame
    p_w = np.ones(4)
    p_w[:3] = point
    p_cam = T_world2cam @ p_w
    x_cam, y_cam, z_cam = p_cam[:3]

    # 3) must be in front of camera
    if z_cam <= 0:
        return False

    # 4) pinhole projection (no distortion since coeffs are zero)
    u = I['fx'] * (x_cam / z_cam) + I['cx']
    v = I['fy'] * (y_cam / z_cam) + I['cy']

    # 5) check image bounds
    in_x = (0.0 <= u) and (u < I['width'])
    in_y = (0.0 <= v) and (v < I['height'])
    return in_x and in_y


def my_nms(boxes, scores, iou_threshold, extra_data_lists=None, three_d=False):
    if extra_data_lists is None:
        extra_data_lists = []
    iou_func = iou_3d if three_d else iou_2d

    # sanity-check extra lists
    n_extra = len(extra_data_lists)
    for i, ed in enumerate(extra_data_lists):
        if len(ed) != len(boxes):
            raise ValueError(f"extra_data_lists[{i}] has length {len(ed)}, "
                             f"but you have {len(boxes)} boxes")

    # pack everything together: (score, box, [extra1, extra2, ...])
    items = []
    for idx, (b, s) in enumerate(zip(boxes, scores)):
        extras = [extra_data_lists[j][idx] for j in range(n_extra)]
        items.append((s, b, extras))

    # sort by score desc
    items.sort(key=lambda x: x[0], reverse=True)

    kept = []
    while items:
        # pop highest‐score
        curr_score, curr_box, curr_extras = items.pop(0)

        if not items:
            # no more to compare → keep as is
            kept.append((curr_score, curr_box, curr_extras))
            break

        # compute IoU vs. all remaining
        ious = np.array([iou_func(curr_box, b2) for (_, b2, _) in items])
        # find which ones exceed threshold
        discard_idx = np.where(ious >= iou_threshold)[0]

        # sum up the *scores* of those to-be-discarded
        discarded_scores = sum(items[i][0] for i in discard_idx)/ len(discard_idx) if discard_idx.size > 0 else 0.0
        curr_score += discarded_scores / 2

        # rebuild items, skipping the discarded indices
        items = [item for i, item in enumerate(items) if i not in discard_idx]

        kept.append((curr_score, curr_box, curr_extras))

    # unzip results
    kept_scores = [s for s, _, _ in kept]
    kept_boxes  = [b for _, b, _ in kept]
    kept_extras = [
        [ext_list[j] for _, _, ext_list in kept]
        for j in range(n_extra)
    ]

    return kept_boxes, kept_scores, kept_extras

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

def parse_gemini_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

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

# --- pcd_to_msg (remains unchanged from previous correct version) ---
def pcd_to_msg(pcd: o3d.geometry.PointCloud, frame_id: str) -> PointCloud2:
    """
    Converts an Open3D PointCloud to a ROS2 PointCloud2 message without ros2_numpy.
    Assumes pcd has points and colors.
    """
    if not pcd.has_points():
        print("Warning: Open3D PointCloud has no points to convert to ROS2 message.")
        return PointCloud2()

    points_np = np.asarray(pcd.points)
    colors_np = np.asarray(pcd.colors)

    rgb_uint8 = (colors_np * 255).astype(np.uint8)

    rgb_packed_uint32 = (
        (rgb_uint8[:, 0].astype(np.uint32) << 16) |
        (rgb_uint8[:, 1].astype(np.uint32) << 8)  |
        (rgb_uint8[:, 2].astype(np.uint32))
    )

    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('rgb', np.float32)
    ])

    pc_data_structured = np.zeros(len(points_np), dtype=dtype)
    pc_data_structured['x'] = points_np[:, 0]
    pc_data_structured['y'] = points_np[:, 1]
    pc_data_structured['z'] = points_np[:, 2]
    pc_data_structured['rgb'] = rgb_packed_uint32.view(np.float32)

    header = Header()
    header.frame_id = frame_id

    fields = []
    fields.append(PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1))
    fields.append(PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1))
    fields.append(PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1))
    fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))

    point_step = pc_data_structured.dtype.itemsize
    row_step = point_step * len(points_np)

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points_np)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = row_step
    msg.is_dense = True
    msg.data = pc_data_structured.tobytes()

    return msg

# --- msg_to_pcd (UPDATED to fix ValueError) ---
def msg_to_pcd(msg: PointCloud2) -> o3d.geometry.PointCloud:
    """
    Inverse of pcd_to_msg. Converts a ROS2 PointCloud2 message to an Open3D PointCloud
    by manually parsing the data field.
    Assumes the PointCloud2 message has 'x', 'y', 'z' and 'rgb' (packed float32) fields.
    """
    pcd = o3d.geometry.PointCloud()

    total_points = msg.width * msg.height
    if total_points == 0:
        print("Warning: Received empty PointCloud2 message.")
        return pcd

    field_info = _get_field_info(msg.fields)

    if not all(f in field_info for f in ['x', 'y', 'z']):
        print("Error: PointCloud2 message missing 'x', 'y', or 'z' fields.")
        return pcd

    xyz_list = []
    rgb_list = []
    has_rgb = 'rgb' in field_info and field_info['rgb']['datatype'] == PointField.FLOAT32

    endian_char = '<' if not msg.is_bigendian else '>'
    format_float32 = endian_char + 'f'

    for i in range(total_points):
        point_start_offset = i * msg.point_step

        x_offset = field_info['x']['offset']
        y_offset = field_info['y']['offset']
        z_offset = field_info['z']['offset']

        x = struct.unpack_from(format_float32, msg.data, point_start_offset + x_offset)[0]
        y = struct.unpack_from(format_float32, msg.data, point_start_offset + y_offset)[0]
        z = struct.unpack_from(format_float32, msg.data, point_start_offset + z_offset)[0]
        xyz_list.append([x, y, z])

        if has_rgb:
            rgb_offset = field_info['rgb']['offset']
            rgb_packed_float32 = struct.unpack_from(format_float32, msg.data, point_start_offset + rgb_offset)[0]

            # --- FIX FOR ValueError: Changing the dtype of a 0d array is only supported if the itemsize is unchanged ---
            # Ensure we operate on a 1D array view, even if it has one element.
            # Convert the scalar float to a 1-element numpy array before viewing.
            rgb_packed_uint32 = np.array([rgb_packed_float32], dtype=np.float32).view(np.uint32)[0]
            # .item() is then called on the single element of the 1D array, or directly use the [0] index
            # This ensures that .view() is called on an array whose itemsize doesn't change from float32 to uint32
            # because the float32 array is 1D.

            r = ((rgb_packed_uint32 >> 16) & 0x0000FF)
            g = ((rgb_packed_uint32 >> 8) & 0x0000FF)
            b = ((rgb_packed_uint32) & 0x0000FF)

            rgb_list.append([r / 255.0, g / 255.0, b / 255.0])

    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz_list))
    if has_rgb and rgb_list:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb_list))
        print("Info: Extracted XYZ and RGB colors from PointCloud2.")
    elif has_rgb and not rgb_list:
        print("Warning: PointCloud2 message had 'rgb' field but no color data was extracted (possibly empty cloud).")
    else:
        print("Warning: PointCloud2 message does not contain a valid 'rgb' field. Only XYZ points extracted.")

    return pcd