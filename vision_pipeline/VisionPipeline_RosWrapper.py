from vision_pipeline.subscriber import RealSenseSubscriber
from vision_pipeline.visionpipeline import VisionPipe
import rclpy
import open3d as o3d

class ROS_VisionPipe(VisionPipe):
    def __init__(self):
        super().__init__()
        self.sub = RealSenseSubscriber("head")
        self.track_strings = []

    def update(self, debug=False):
        rgb_img = self.sub.latest_rgb
        depth_img = self.sub.latest_depth
        intrinsics = self.sub.get_intrinsics()
        pose = [0,0,0,0,0,0]
        if rgb_img is None or depth_img is None or intrinsics is None:
            print("No image received yet.")
            return False
        if len(self.track_strings) == 0:
            print("No track strings provided.")
            return False

        return super().update(rgb_img, depth_img, self.track_strings, intrinsics, pose, debug=debug)

    def add_track_string(self, new_track_string):
        if isinstance(new_track_string, str) and new_track_string not in self.track_strings:
            self.track_strings.append(new_track_string)
        elif isinstance(new_track_string, list):
            [self.track_strings.append(x) for x in new_track_string if x not in self.track_strings]
        else:
            raise ValueError("track_string must be a string or a list of strings")


def main(args=None):
    rclpy.init(args=args)
    VP = ROS_VisionPipe()
    VP.add_track_string("drill")
    VP.add_track_string(["wrench", "screwdriver"])
    success_counter = 0
    while success_counter < 5:
        success = VP.update()
        success_counter += 1 if success != False else 0

    print("Success counter: ", success_counter)

    for object, predictions in VP.tracked_objects.items():
        print(f"{object=}")
        print(f"   {len(predictions['boxes'])=}, {len(predictions['pcds'])=}, {predictions['scores'].shape=}")
        for i, pcd in enumerate(predictions["pcds"]):
            print(f"   {i=}, {predictions['scores'][i]=}")
    print("")

    VP.display()
if __name__ == "__main__":
    main()