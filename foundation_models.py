import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import warnings
import matplotlib.pyplot as plt
import cv2
import numpy as np
from capture_cameras import get_cap, RealSenseCamera
import open3d as o3d
import torch.nn.functional as F


import numpy as np
from torchvision.ops import clip_boxes_to_image
from utils import get_points_and_colors, nms

class OWLv2:
    def __init__(self, iou_th=0.25, discard_percentile = 0.5):
        """
        Initializes the OWLv2 model and processor.
        Parameters:
        - iou_th: IoU threshold for NMS
        - discard_percentile: percentile to discard low scores
        """
        # Load the OWLv2 model and processor
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

        # Set device to GPU if available, otherwise CPU
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else self.device

        # Move model to the appropriate device
        self.model.to(self.device)
        self.model.eval()  # set model to evaluation mode

        # Set the IoU threshold and discard percentile
        self.iou_th = iou_th
        self.discard_percentile = discard_percentile
    def predict(self, img, querries, debug = False):
        """
        Gets realsense frames
        Parameters:
        - img: image to produce bounding boxes in
        - querries: list of strings whos bounding boxes we want
        - debug: if True, prints debug information
        Returns:
        - out_dict: dictionary containing a list of bounding boxes and a list of scores for each querry
        """
        #Preprocess inputs
        inputs = self.processor(text=querries, images=img, return_tensors="pt")
        inputs.to(self.device)

        #model forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([img.shape[:2]])  # (height, width)

        results = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0)[0]

        # Extract labels, boxes, and scores
        all_labels = results["labels"]
        all_boxes = results["boxes"]
        all_boxes = clip_boxes_to_image(all_boxes, img.shape[:2])
        all_scores = results["scores"]

        #get integer to text label mapping
        label_lookup = {i: label for i, label in enumerate(querries)}
        out_dict = {}
        #for each querry, get the boxes and scores and perform NMS
        for i, label in enumerate(querries):
            text_label = label_lookup[i]

            # Filter boxes and scores for the current label
            mask = all_labels == i
            instance_boxes = all_boxes[mask]
            instance_scores = all_scores[mask]

            #Do NMS for the current label
            keep = nms(instance_boxes.cpu(), instance_scores.cpu(), iou_threshold=self.iou_th)
            pruned_boxes  = instance_boxes[keep]
            pruned_scores = instance_scores[keep]

            #Get rid of low scores
            threshold = torch.quantile(pruned_scores, self.discard_percentile)
            keep = pruned_scores > threshold
            filtered_scores = pruned_scores[keep]

            # Normalize scores
            filtered_scores = filtered_scores / filtered_scores.sum()
            filtered_boxes  = pruned_boxes[keep]

            # Update output dictionary
            out_dict[text_label] = {"scores": filtered_scores, "boxes": filtered_boxes}

            if debug:
                print(f"{text_label=}")
                print(f"    {all_labels.shape=}, {all_boxes.shape=}, {all_scores.shape=}")
                print(f"    {instance_scores.shape=}, {instance_boxes.shape=}")
                print(f"    {pruned_boxes.shape=}, {pruned_scores.shape=}")
                print(f"    {len(keep)=}")
                print(f"    {filtered_scores.shape=}, {filtered_boxes.shape=}")
                print(f"    {threshold=}")
                print()


        if debug:
            for key in out_dict:
                print(f"{key=} {out_dict[key]['boxes'].shape=}, {out_dict[key]['scores'].shape=}")
        return out_dict

    def __str__(self):
        return f"OWLv2: {self.model.device}"
    def __repr__(self):
        return self.__str__()
def test_OWL(left_img):
    
    owl = OWLv2()
    predicitons = owl.predict(left_img, ["phone", "water bottle", "can"], debug=True)
    #print(f"{predicitons=}")
    for querry_object, prediction in predicitons.items():
        display_img = left_img.copy()
        for bbox, score in zip(prediction["boxes"], prediction["scores"]):
            #bbox = prediction["box"]
            #score = prediction["score"]
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(display_img, f"{querry_object} {score:.4f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Display the image with bounding boxes
        cv2.imshow(f"{querry_object}", display_img)
        cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return predicitons
#Class to use sam2
class SAM2_PC:
    def __init__(self, iou_th=0.1):
        """
        Initializes the SAM2 model and processor.
        Parameters:
        - iou_th: IoU threshold for NMS
        """
        self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else self.device
        self.iou_th = iou_th
    def predict(self, rgb_img, depth_img, bbox, scores, intrinsics, debug = False):
        """
        Predicts 3D point clouds from RGB and depth images and bounding boxes using SAM2.
        Cleans up the point clouds and applies NMS.
        Parameters:
        - rgb_img: RGB image
        - depth_img: Depth image
        - bbox: Bounding boxes
        - scores: Scores for the bounding boxes
        - intrinsics: Camera intrinsics
        - debug: If True, prints debug information
        Returns:
        - reduced_pcds: List of reduced point clouds
        - reduced_bounding_boxes_3d: List of reduced 3D bounding boxes
        - reduced_scores: List of reduced scores
        """
        #Run sam2 on all the boxes
        self.sam_predictor.set_image(rgb_img)
        sam_mask = None
        sam_scores = None
        sam_logits = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            sam_mask, sam_scores, sam_logits = self.sam_predictor.predict(box=bbox)
        sam_mask = np.all(sam_mask, axis=1)
        if debug:
            print(f"{sam_mask.shape=}")


        #Apply mask to the depth and rgb images
        masked_depth = depth_img[None, ...] * sam_mask
        masked_rgb = rgb_img[None, ...] * sam_mask[..., None]
        if debug:
            print(f"{masked_depth.shape=}, {masked_rgb.shape=}")
            fig, ax = plt.subplots(5, 3)
            for i in range(min(5, sam_mask.shape[0])):
                bbox_img = rgb_img.copy()
                x_min, y_min, x_max, y_max = map(int, bbox[i])
                cv2.rectangle(bbox_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                ax[i, 0].imshow(bbox_img)
                ax[i, 1].imshow(masked_rgb[i])
                ax[i, 2].imshow(masked_depth[i])
            fig.tight_layout()
            plt.show()
        
        #Get points and colors from masked depth and rgb images
        tensor_depth = torch.from_numpy(masked_depth).to(self.device)
        tensor_rgb = torch.from_numpy(masked_rgb).to(self.device)
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        points, colors = get_points_and_colors(tensor_depth, tensor_rgb, fx, fy, cx, cy)
        colors = colors[..., [2, 1, 0]]
        if debug:
            print(f"{points.shape=}, {colors.shape=}")

        B, N, _ = points.shape
        full_pcds = []
        full_bounding_boxes_3d = []
        full_scores = []
        pts_cpu   = points.detach().cpu()
        cols_cpu  = colors.detach().cpu()
        #for each candiate object get the point cloud unless there are too few points
        for i in range(B):
            pts = pts_cpu[i]          # (N,3)
            cls = cols_cpu[i]         # (N,3)

            # mask out void points
            valid = pts[:, 2] > 0
            if valid.sum() <  100:
                continue
            pts_valid = pts[valid]
            cls_valid = cls[valid]

            # build Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_valid.numpy())
            pcd.colors = o3d.utility.Vector3dVector(cls_valid.numpy()/255)
            # Apply statistical outlier removal to denoise the point cloud
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            bbox = pcd.get_axis_aligned_bounding_box()
            bbox.color = (1.0, 0.0, 0.0)  # red 
            full_pcds.append(pcd)
            full_scores.append(scores[i])
            full_bounding_boxes_3d.append(bbox)

        # Perform NMS on the 3D bounding boxes
        keep = nms(full_bounding_boxes_3d, full_scores, iou_threshold=self.iou_th, three_d=True)
        
        # Filter the point clouds, bounding boxes, and scores based on NMS results
        reduced_pcds = [pcd for i, pcd in enumerate(full_pcds) if i in keep]
        reduced_bounding_boxes_3d = [bbox for i, bbox in enumerate(full_bounding_boxes_3d) if i in keep]
        reduced_scores = [score for i, score in enumerate(full_scores) if i in keep]
        
        # Normalize the scores
        reduced_scores = torch.tensor(reduced_scores)
        reduced_scores = reduced_scores / reduced_scores.sum()
        if debug:
            print(f"   {len(full_pcds)=}, {len(full_scores)=}, {len(full_bounding_boxes_3d)=}")
            print(f"   {len(reduced_pcds)=}, {len(reduced_bounding_boxes_3d)=}, {reduced_scores.shape=}")

        
        return reduced_pcds, reduced_bounding_boxes_3d, reduced_scores
    def __str__(self):
        return f"SAM2: {self.sam_predictor.model.device}"
    def __repr__(self):
        return self.__str__()
def test_sam(rgb_img, depth_img, predictions, intrinsics):
    sam = SAM2_PC()
    for querry_object, canditates in predictions.items():
        print("\n\n")
        point_clouds, boxes, scores = sam.predict(rgb_img, depth_img, canditates["boxes"], canditates["scores"], intrinsics, debug=True)
            
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2,    # e.g. half a meter
            origin=[0, 0, 0]
        )

        # 2) Start your geometry list with that axis
        geoms = [camera_frame]
        geoms += point_clouds
        geoms += boxes



        o3d.visualization.draw_geometries(geoms, window_name=f"{querry_object} PointClouds")
    #print(f"{len(point_clouds)=}, {len(boxes)=}")
    return None



if __name__=="__main__":
    cap = RealSenseCamera()
    I = cap.get_intrinsics()
    ret, rgb_img, depth_img = cap.read(return_depth=True)
    if not ret:
        print("Error: Unable to read frame from the camera.")
        exit(1)
    
    print(f"\n\nTESTING OWL")
    predictions = test_OWL(rgb_img)
    
    print(f"\n\nTESTING SAM")
    test_sam(rgb_img, depth_img, predictions, I)    



    

