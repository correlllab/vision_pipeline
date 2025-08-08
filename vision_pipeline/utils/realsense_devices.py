import pyrealsense2 as rs
import numpy as np
import time
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d.core as o3c
import open3d.t.geometry as o3tg

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
        self.open = True

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
        self.open = False


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



# --- Main Application ---
def main():
    import threading
    import numpy as np
    import matplotlib
    matplotlib.use("Qt5Agg", force=True)  # avoid backend fights
    import matplotlib.pyplot as plt

    running = True
    lock = threading.Lock()
    latest_vertices = None
    latest_colors = None
    MAX_POINTS = 300_000
    TICK_PERIOD = 0.05  # ~20Hz GUI updates; tune as needed

    # ---- Open3D GUI setup (main thread) ----
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("RealSense 3D Viewer", 1024, 768)
    vis.show_settings = True

    # device = o3c.Device("CUDA:0")
    device = o3c.Device("CPU:0")

    # Preallocate a fixed-size point cloud
    positions_np = np.zeros((MAX_POINTS, 3), dtype=np.float32)
    colors_np    = np.zeros((MAX_POINTS, 3), dtype=np.float32)

    pcd = o3tg.PointCloud(device)
    pcd.point["positions"] = o3c.Tensor(positions_np, o3c.float32, device)
    pcd.point["colors"]    = o3c.Tensor(colors_np,   o3c.float32, device)

    vis.add_geometry("pcd", pcd)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry("world_axis", coord_frame)
    vis.reset_camera_to_default()
    app.add_window(vis)

    # Live timing plot (in-process but not on GUI thread)
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line1, = ax.plot([], [], label='1. Get Data')
    line2, = ax.plot([], [], label='2. Update Tensor')
    line3, = ax.plot([], [], label='3. Update Geometry')
    line4, = ax.plot([], [], label='4. Pump Callback')
    ax.set_xlabel('Frame Number'); ax.set_ylabel('Time (seconds)')
    ax.set_title('RealSense Data Processing Times Per Frame (Live)')
    ax.grid(True); ax.legend()
    acc_1, acc_2, acc_3, acc_4 = [], [], [], []
    last_plot = time.time()

    def update_plot(t1, t2, t3, t4):
        nonlocal last_plot
        acc_1.append(t1); acc_2.append(t2); acc_3.append(t3); acc_4.append(t4)
        # Throttle plotting a bit
        if time.time() - last_plot > 0.05 and plt.fignum_exists(fig.number):
            x = np.arange(1, len(acc_1) + 1)
            line1.set_data(x, acc_1); line2.set_data(x, acc_2)
            line3.set_data(x, acc_3); line4.set_data(x, acc_4)
            ax.relim(); ax.autoscale_view()
            fig.canvas.draw_idle(); fig.canvas.flush_events()
            last_plot = time.time()

    # ---- RealSense grabber thread (no GUI calls here) ----
    def grabber():
        nonlocal latest_vertices, latest_colors, running
        cam = None
        try:
            cam = RealSenseCamera()
            # short warmup
            for _ in range(10):
                cam.get_data()
                time.sleep(0.02)

            while running:
                s = time.time()
                data = cam.get_data()  # make sure this has a short internal timeout
                t_get = time.time() - s

                with lock:
                    latest_vertices = data["vertices"]
                    latest_colors   = data["colors"]

                # tiny nap to avoid maxing a CPU
                time.sleep(0.001)
        except Exception as e:
            print("Grabber error:", e)
        finally:
            if cam:
                try: cam.stop()
                except Exception: pass

    # ---- Pump: called on the Qt main thread via post_to_main_thread ----
    def pump():
        # Copy references quickly while holding the lock
        with lock:
            verts = latest_vertices
            cols  = latest_colors

        if verts is None or cols is None:
            return  # nothing yet

        s2 = time.time()
        n = min(len(verts), MAX_POINTS)
        if n > 0:
            # slice-assign into preallocated tensors (no reallocation)
            pcd.point["positions"][:n] = o3c.Tensor(verts[:n], o3c.float32, device)
            pcd.point["colors"][:n]    = o3c.Tensor(cols[:n],  o3c.float32, device)
        if n < MAX_POINTS:
            # avoid NaNs; fill the tail with zeros
            pcd.point["positions"][n:] = 0
            pcd.point["colors"][n:]    = 0
        t_upd = time.time() - s2

        s3 = time.time()
        flags = (rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG)
        vis.update_geometry("pcd", pcd, flags)
        vis.post_redraw()
        t_geo = time.time() - s3

        # Record “pump” cost as t_gui
        t_gui = t_upd + t_geo
        # We can’t measure t_get here; approximate from last grabber tick if you expose it.
        update_plot(0.0, t_upd, t_geo, t_gui)

    # ---- Ticker thread: periodically schedule pump() on the GUI thread ----
    def ticker():
        while running:
            try:
                app.post_to_main_thread(vis, pump)
            except Exception:
                pass
            time.sleep(TICK_PERIOD)

    # Clean shutdown when user closes the Open3D window
    def on_close():
        nonlocal running
        running = False
        return True
    vis.set_on_close(on_close)

    # Start threads
    th_grab = threading.Thread(target=grabber, daemon=True)
    th_tick = threading.Thread(target=ticker,  daemon=True)
    th_grab.start(); th_tick.start()

    # Hand control to Qt (single, authoritative GUI loop)
    try:
        app.run()
    finally:
        # Stop threads and close plot
        running = False
        th_grab.join(timeout=2)
        th_tick.join(timeout=2)
        plt.ioff()
        try: plt.close(fig)
        except Exception: pass

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    # rgb_color_img = cv2.cvtColor(data["color_img"], cv2.COLOR_BGR2RGB)
    # o3d_color = o3d.geometry.Image(rgb_color_img)
    # o3d_depth = o3d.geometry.Image(data["depth_img"])
    # rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth, convert_rgb_to_intensity=False, depth_scale=1.0/camera.depth_scale)
    # rgbd_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    #         width=data['color_intrinsics'].width,
    #         height=data['color_intrinsics'].height,
    #         fx=data['color_intrinsics'].fx,
    #         fy=data['color_intrinsics'].fy,
    #         cx=data['color_intrinsics'].ppx,
    #         cy=data['color_intrinsics'].ppy
    #     )
    # rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, rgbd_intrinsics)

    # rs_pcd = o3d.geometry.PointCloud()
    # rs_pcd.points = o3d.utility.Vector3dVector(data['vertices'])
    # rs_pcd.colors = o3d.utility.Vector3dVector(data['colors'])

    
    # print(f"Number of points in RealSense-generated PCD: {len(rs_pcd.points)}")
    # print(f"Number of points in Open3D-generated PCD:   {len(rgbd_pcd.points)}")
    # print("-" * 30)

    # # Calculate the distance from each point in rs_pcd to the nearest point in rgbd_pcd
    # distance = rs_pcd.compute_point_cloud_distance(rgbd_pcd)
    # distance_array = np.asarray(distance)

    # if len(distance_array) > 0:
    #     print("Point-to-point distance statistics:")
    #     print(f"  Mean distance: {np.mean(distance_array):.6f} meters")
    #     print(f"  Max distance:  {np.max(distance_array):.6f} meters")
    #     print(f"  Std. dev.:     {np.std(distance_array):.6f} meters")
    # else:
    #     print("Could not compute distance, one of the point clouds is empty.")

    
    # rs_common, rgbd_common = pointcloud_intersection(rs_pcd, rgbd_pcd, radius=0.005)

    # # Color the RealSense-generated cloud RED
    # rs_pcd.paint_uniform_color([1, 0, 0])

    # # Color the Open3D-generated cloud BLUE
    # rgbd_pcd.paint_uniform_color([0, 0, 1])
    
    # # Visualize them together
    # o3d.visualization.draw_geometries([rgbd_pcd, rs_pcd])

    # print(f"Found {len(rs_common.points)} points of rs_pcd in rgbd_pcd")
    # print(f"Found {len(rgbd_common.points)} points of rgbd_pcd in rs_pcd")

    # o3d.visualization.draw_geometries(
    #     [rs_common],
    #     window_name="points of rs_pcd in rgbd_pcd"
    # )

    # o3d.visualization.draw_geometries(
    #     [rgbd_common],
    #     window_name=" points of rgbd_pcd in rs_pcd"
    # )



    pass
