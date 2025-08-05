import pyrealsense2 as rs
import cv2
import open3d as o3d
import numpy as np
import time
def list_realsense_devices():
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

class RealSenseCamera:
    """
    Manages a RealSense camera, providing a method to get colored point clouds
    and calibration data, including IMU extrinsics if available.
    """
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable standard streams
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        print("Starting RealSense pipeline...")
        profile = self.pipeline.start(config)
        
        # Get static calibration data
        self.get_calibration_data(profile)
        
        # Initialize processing blocks
        self.align = rs.align(rs.stream.color)
        self.pointcloud = rs.pointcloud()

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

    def get_calibration_data(self, profile):
        """Fetches and stores all static calibration data."""
        # Intrinsics
        self.color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        # Extrinsics between camera sensors
        self.extrinsics_depth_to_color = profile.get_stream(rs.stream.depth).get_extrinsics_to(profile.get_stream(rs.stream.color))
        
        
    def get_data(self):
        """
        Gets a snapshot of data, including a colored point cloud.
        """
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None
        
        # --- Create Colored Point Cloud ---
        # 1. Calculate the 3D points (vertices)
        self.pointcloud.map_to(color_frame)  
        points = self.pointcloud.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices(2))
        
        # 2. Get the corresponding color for each point
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 3. Reshape and filter out invalid points
        # Reshape data into lists of points and colors
        vertices = vertices.reshape(-1, 3)
        colors = color_image.reshape(-1, 3)
        
        # Filter out points where depth is 0 (vertices are [0,0,0])
        valid_points_mask = vertices.any(axis=1)
        valid_vertices = vertices[valid_points_mask]
        valid_colors = colors[valid_points_mask]
        
        # --- Package all data into a dictionary ---
        data = {
            "color_img": color_image,
            "depth_img": depth_image,
            "vertices": valid_vertices,
            "colors": valid_colors[:, ::-1]/255, # BGR format
            "color_intrinsics": self.color_intrinsics,
            "extrinsics_depth_to_color": self.extrinsics_depth_to_color
        }
        
            
        return data

    def stop(self):
        """Stops the pipeline."""
        print("Stopping RealSense pipeline...")
        self.pipeline.stop()


def pointcloud_intersection(pcd1, pcd2, radius=0.005):
    """
    Returns two new point clouds:
      - in1: points from pcd1 that have a neighbor in pcd2 within `radius`
      - in2: points from pcd2 that have a neighbor in pcd1 within `radius`
    """
    # Build KD‐trees
    kdtree1 = o3d.geometry.KDTreeFlann(pcd1)
    kdtree2 = o3d.geometry.KDTreeFlann(pcd2)
    
    # Find pcd1 points in pcd2
    idxs1 = []
    pts1 = np.asarray(pcd1.points)
    for i, p in enumerate(pts1):
        [cnt, idx, _] = kdtree2.search_radius_vector_3d(p, radius)
        if cnt > 0:
            idxs1.append(i)
    
    # Find pcd2 points in pcd1
    idxs2 = []
    pts2 = np.asarray(pcd2.points)
    for i, p in enumerate(pts2):
        [cnt, idx, _] = kdtree1.search_radius_vector_3d(p, radius)
        if cnt > 0:
            idxs2.append(i)
    
    # Select matching points
    in1 = pcd1.select_by_index(idxs1)
    in2 = pcd2.select_by_index(idxs2)
    return in1, in2


if __name__ == "__main__":
    camera = None
    data = None

    # Create the visualizer and window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='PCD Animation')

    # Add the geometry to the scene ONCE before the loop
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
    vis.add_geometry(coord_frame)

    try:
        # Initialize the camera
        camera = RealSenseCamera()
        data = camera.get_data()


        exit = False
        while not exit:
            # Get data from the camera
            data = camera.get_data()
            if not data:
                continue

            # --- OpenCV Windows ---
            cv2.imshow("Color", data["color_img"])
            d_norm = cv2.normalize(data["depth_img"], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow("Depth", d_norm)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit = True
            
            # print(f"{data['vertices'].shape=}")
            pcd.points = o3d.utility.Vector3dVector(data['vertices'])
            pcd.colors = o3d.utility.Vector3dVector(data['colors'])
            
            # Tell the visualizer that the pcd object has been updated
            pcd.voxel_down_sample(voxel_size=0.01)
            vis.update_geometry(pcd)
            
            # Process window events and redraw the scene
            vis.poll_events()
            vis.update_renderer()
           
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up...")
        if camera:
            camera.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()







    rgb_color_img = cv2.cvtColor(data["color_img"], cv2.COLOR_BGR2RGB)
    o3d_color = o3d.geometry.Image(rgb_color_img)
    o3d_depth = o3d.geometry.Image(data["depth_img"])
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth, convert_rgb_to_intensity=False, depth_scale=1.0/camera.depth_scale)
    rgbd_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=data['color_intrinsics'].width,
            height=data['color_intrinsics'].height,
            fx=data['color_intrinsics'].fx,
            fy=data['color_intrinsics'].fy,
            cx=data['color_intrinsics'].ppx,
            cy=data['color_intrinsics'].ppy
        )
    rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, rgbd_intrinsics)

    rs_pcd = o3d.geometry.PointCloud()
    rs_pcd.points = o3d.utility.Vector3dVector(data['vertices'])
    rs_pcd.colors = o3d.utility.Vector3dVector(data['colors'])

    
    print(f"Number of points in RealSense-generated PCD: {len(rs_pcd.points)}")
    print(f"Number of points in Open3D-generated PCD:   {len(rgbd_pcd.points)}")
    print("-" * 30)

    # Calculate the distance from each point in rs_pcd to the nearest point in rgbd_pcd
    distance = rs_pcd.compute_point_cloud_distance(rgbd_pcd)
    distance_array = np.asarray(distance)

    if len(distance_array) > 0:
        print("Point-to-point distance statistics:")
        print(f"  Mean distance: {np.mean(distance_array):.6f} meters")
        print(f"  Max distance:  {np.max(distance_array):.6f} meters")
        print(f"  Std. dev.:     {np.std(distance_array):.6f} meters")
    else:
        print("Could not compute distance, one of the point clouds is empty.")

    
    rs_common, rgbd_common = pointcloud_intersection(rs_pcd, rgbd_pcd, radius=0.005)

    # Color the RealSense-generated cloud RED
    rs_pcd.paint_uniform_color([1, 0, 0])

    # Color the Open3D-generated cloud BLUE
    rgbd_pcd.paint_uniform_color([0, 0, 1])
    
    # Visualize them together
    o3d.visualization.draw_geometries([rgbd_pcd, rs_pcd])

    print(f"Found {len(rs_common.points)} points of rs_pcd in rgbd_pcd")
    print(f"Found {len(rgbd_common.points)} points of rgbd_pcd in rs_pcd")

    o3d.visualization.draw_geometries(
        [rs_common],
        window_name="points of rs_pcd in rgbd_pcd"
    )

    o3d.visualization.draw_geometries(
        [rgbd_common],
        window_name=" points of rgbd_pcd in rs_pcd"
    )



    

