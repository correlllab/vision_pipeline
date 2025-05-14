from foundation_models import OWLv2, SAM2_PC

class VisionPipe:
    def __init__(self):
        self.owv2 = OWLv2()
        self.sam2 = SAM2_PC()
        self.tracked_objects = {}
    def update(self, rgb_img, depth_img, querries, I, debug = True):
        predictions_2d = self.owv2.predict(rgb_img, querries)
        predictions_3d = {}
        for object, prediction_2d in predictions_2d.items():
            pcds, box_3d, scores = self.sam2.predict(rgb_img, depth_img, prediction_2d["boxes"], prediction_2d["scores"], I)
            predictions_3d[object] = {"boxes": box_3d, "scores": scores, "pcds": pcds}
            if debug:
                print(f"{object=}")
                print(f"   {predictions_2d[object]['boxes'].shape=}, {predictions_2d[object]['scores'].shape=}")
                print(f"   {len(predictions_3d[object]['boxes'])=}, {len(predictions_3d[object]['pcds'])=}, {predictions_3d[object]['scores'].shape=}")

        #HELP ME UPDATE MY BELIEFES IN TRACKED OBJECTS

    def querry(self, querry):
        # Help me Get the highest belife object for the querry
        pass



def test_VP(cap):
    vp = VisionPipe()
    for i in range(5):
        ret, rgb_img, depth_img = cap.read(return_depth=True)
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break
        I = cap.get_intrinsics()
        predictions = vp.update(rgb_img, depth_img, ["phone", "water bottle"], I)
        
            
        print(f"\n\n")


if __name__ == "__main__":
    from capture_cameras import RealSenseCamera
    cap = RealSenseCamera()
    I = cap.get_intrinsics()
    ret, rgb_img, depth_img = cap.read(return_depth=True)
    if not ret:
        print("Error: Unable to read frame from the camera.")
        exit(1)

    print(f"\n\nTESTING VP")
    test_VP(cap)