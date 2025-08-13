import pyrealsense2 as rs
import numpy as np
import time
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d.core as o3c
import open3d.t.geometry as o3tg
import threading, time, traceback


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


def vis_loop(grabber_function, display_world = True, window_name ="RealSense 3D Viewer", provide_cam = True):
    """
    grabber function must implement a single iteration in which it consume the data from RealSenseCamera.get_data + the lock and sets latest_pcs, latest_bbs, latest_probs
    """
    # Display Variables 
    running = True
    lock = threading.Lock()
    latest_pcs = {} #used for objects
    latest_bbs = {} #used for objects
    latest_probs = {} #used for objects
    latest_vertices = [] #used for world
    latest_colors = [] #used for world

    TICK_PERIOD = 0.05


    # Open3D GUI setup 
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer(f"{window_name}", 1024, 768)
    vis.show_settings = True
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    vis.add_geometry("world_axis", coord_frame)

    if display_world:
        WORLD_MAX_POINTS = 300_000
        positions_np = np.zeros((WORLD_MAX_POINTS, 3), dtype=np.float32)
        colors_np = np.zeros((WORLD_MAX_POINTS, 3), dtype=np.float32)
        device = o3d.core.Device("CPU:0")
        world_pc = o3d.t.geometry.PointCloud()
        world_pc.point["positions"] = o3d.core.Tensor(positions_np, o3d.core.float32, device)
        world_pc.point["colors"]    = o3d.core.Tensor(colors_np,   o3d.core.float32, device)
        vis.add_geometry("world_pc", world_pc)
        
    vis.reset_camera_to_default()
    app.add_window(vis)

    last_frame = set()

    def pump():
        pcs = None
        bbs = None
        with lock:
            pcs   = {k: v[:] for k, v in latest_pcs.items()}
            bbs   = {k: v[:] for k, v in latest_bbs.items()}
            probs = {k: v[:] for k, v in latest_probs.items()}
            if display_world:
                verts = latest_vertices
                colors = latest_colors

        
        flags = (rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG)
        nonlocal last_frame
        this_frame = set()

        vis.clear_3d_labels()
        for q in pcs.keys():
            if q not in pcs or q not in bbs:
                continue
            q_pcs = pcs[q]
            q_bboxs = bbs[q]
            q_probs = probs[q]
            assert len(q_pcs) == len(q_bboxs)
            assert len(q_pcs) == len(q_probs)

            for i in range(len(q_pcs)):
                pcd = q_pcs[i]
                bbox = q_bboxs[i]
                prob = q_probs[i]

                # Point cloud
                name_pcd = f"{q.replace(' ', '_')}_pcd_{i}"
                if name_pcd in last_frame:
                    vis.remove_geometry(name_pcd)
                vis.add_geometry(name_pcd, pcd) 
                this_frame.add(name_pcd)
                

                # # Bounding box
                name_bbox = f"{q.replace(' ', '_')}_bbox_{i}"
                if name_bbox in last_frame:
                    vis.remove_geometry(name_bbox)
                legacy_bbox = bbox.to_legacy()
                legacy_bbox.color = (1-prob, prob, 0)
                vis.add_geometry(name_bbox, legacy_bbox)
                this_frame.add(name_bbox)

                vis.add_3d_label(pcd.get_center().numpy(), f"{q} {prob:.2f}")


        for stale_geometry in (last_frame - this_frame):
            vis.remove_geometry(stale_geometry)
            
        last_frame = this_frame

        #print(f"{display_world=}, {len(verts)=}, {len(colors)=} {len(verts) == len(colors)}")
        if display_world and len(verts) > 0 and len(colors) > 0 and len(verts) == len(colors):
            #print("world updated")
            n = min(len(verts), WORLD_MAX_POINTS)
            if n > 0:
                # slice-assign into preallocated tensors (no reallocation)
                world_pc.point["positions"][:n] = o3d.core.Tensor(verts[:n],  o3d.core.float32, device)
                world_pc.point["colors"][:n]    =  o3d.core.Tensor(colors[:n],   o3d.core.float32, device)
            if n < WORLD_MAX_POINTS:
                # avoid NaNs; fill the tail with zeros
                world_pc.point["positions"][n:] = 0
                world_pc.point["colors"][n:]    = 0
            flags = (rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG)
            vis.update_geometry("world_pc", world_pc, flags)
            vis.post_redraw()

        vis.post_redraw()

    def ticker():
        while running:
            try:
                app.post_to_main_thread(vis, pump)
            except Exception:
                pass
            time.sleep(TICK_PERIOD)

    def on_close():
        nonlocal running
        running = False
        return True
    
    def grabber_wrapper():
        nonlocal latest_pcs, latest_bbs, latest_probs, running
        if display_world:
            nonlocal latest_vertices, latest_colors
        cam = None
        try:
            if provide_cam:
                cam = RealSenseCamera()
                for _ in range(10):
                    cam.get_data()
                    time.sleep(0.02)
            while running:
                grabber_out = grabber_function(cam)
                with lock:
                    latest_vertices = grabber_out["vertices"]
                    latest_colors = grabber_out["colors"]
                    latest_pcs = grabber_out["pcs"]
                    latest_bbs = grabber_out["bbs"]
                    latest_probs = grabber_out["probs"]
                time.sleep(0.001)
        except Exception as e:
            print("Grabber error:", e)
            traceback.print_exc()
        finally:
            if cam:
                try: cam.stop()
                except Exception: pass


    vis.set_on_close(on_close)

    # Start threads
    th_grab = threading.Thread(target=grabber_wrapper, daemon=True)
    th_tick = threading.Thread(target=ticker,  daemon=True)
    th_grab.start(); th_tick.start()

    try:
        app.run()
    finally:
        # Stop threads and close plot
        running = False
        th_grab.join(timeout=2)
        th_tick.join(timeout=2)


if __name__ == "__main__":
    def grabber_func(camera):
        data = camera.get_data()  # make sure this has a short internal timeout
        data["pcs"] = {}
        data["bbs"] = {}
        data["probs"] = {}
        return data
    vis_loop(grabber_func)
  