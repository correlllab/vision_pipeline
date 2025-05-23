from vision_pipeline.subscriber import RealSenseSubscriber
from vision_pipeline.foundation_models import OWLv2, SAM2_PC, display_owl, display_sam2
import rclpy


def main(args=None):
    rclpy.init(args=args)
    sub = RealSenseSubscriber("head")
    OWL = OWLv2()
    SAM = SAM2_PC()
    while True:
        rgb_img = sub.latest_rgb
        depth_img = sub.latest_depth
        print("RGB img shape: ", rgb_img.shape)
        print("Depth img shape: ", depth_img.shape)
        intrinsics = sub.get_intrinsics()
        obs_pose = [0, 0, 0, 0, 0, 0]
        if rgb_img is None or depth_img is None or intrinsics is None:
            print("Waiting for images...")
            continue
        print("RGB img shape: ", rgb_img.shape)
        print("Depth img shape: ", depth_img.shape)
        querries = ["drill", "screw driver", "wrench"]
        predictions_2d = OWL.predict(rgb_img, querries, debug=False)
        for querry_object, canditates in predictions_2d.items():
            print("\n\n")
            point_clouds, boxes, scores = SAM.predict(rgb_img, depth_img, canditates["boxes"], canditates["scores"], intrinsics, debug=False)
if __name__ == "__main__":
    main()