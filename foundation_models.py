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
from torchvision.ops import box_iou, nms, batched_nms, clip_boxes_to_image


class OWLv2:
    def __init__(self, iou_th=0.25, discard_percentile = 0.5):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

        self.model.to(torch.device("cuda")) if torch.cuda.is_available() else None
        self.model.to(torch.device("mps")) if torch.backends.mps.is_available() else None
        self.model.eval()  # set model to evaluation mode

        self.iou_th = iou_th
        self.discard_percentile = discard_percentile
    def predict(self, img, querries, debug = False):
        """
        Gets realsense frames
        Parameters:
        - img: image to produce bounding boxes in
        - querries: list of strings whos bounding boxes we want

        Returns:
        - highest_score_boxes: list of bounding boxes associated with querries
        """
        inputs = self.processor(text=querries, images=img, return_tensors="pt")
        inputs.to(torch.device("cuda")) if torch.cuda.is_available() else None
        inputs.to(torch.device("mps")) if torch.backends.mps.is_available() else None

        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([img.shape[:2]])  # (height, width)

        results = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0)[0]
        # print(f"\n\n{results}\n\n")
        #scores = F.softmax(scores, dim=0)
        # print(f"{scores.sum()}")  # should print tensor(1., device='cuda:0')
        # print(f"{scores=}")
        # print(f"{scores.max()=}")

        all_labels = results["labels"]
        all_boxes = results["boxes"]
        all_boxes = clip_boxes_to_image(all_boxes, img.shape[:2])
        all_scores = results["scores"]
        
        keep = batched_nms(all_boxes, all_scores, all_labels, iou_threshold=self.iou_th)
        pruned_boxes  = all_boxes[keep]
        pruned_scores = all_scores[keep]
        pruned_labels = all_labels[keep]
        if debug:
            print(f"{all_labels.shape=}, {pruned_labels.shape=}")
            print(f"{all_scores.shape=}, {pruned_scores.shape=}")
            print(f"{all_boxes.shape=}, {pruned_boxes.shape=}")

        label_lookup = {i: label for i, label in enumerate(querries)}
        out_dict = {}
        for i, label in enumerate(querries):
            text_label = label_lookup[i]
            mask = pruned_labels == i

            instance_boxes = pruned_boxes[mask]
            instance_scores = pruned_scores[mask]

            threshold = torch.quantile(instance_scores, self.discard_percentile)
            keep = instance_scores > threshold
            filtered_scores = instance_scores[keep]
            filtered_scores = filtered_scores / filtered_scores.sum()
            filtered_boxes  = instance_boxes[keep]

            if debug:
                print(f"{text_label=}")
                print(f"    {instance_scores.shape=}, {instance_boxes.shape=}")
                print(f"    {filtered_scores.shape=}, {filtered_boxes.shape=}")
                print(f"    {threshold=}")
                print()

            out_dict[text_label] = {"scores": filtered_scores, "boxes": filtered_boxes}

        if debug:
            for key in out_dict:
                print(f"{key=} {out_dict[key]['boxes'].shape=}, {out_dict[key]['scores'].shape=}")
        return out_dict

    def __str__(self):
        return f"OWLv2: {self.model.device}"
    def __repr__(self):
        return self.__str__()


def batch_backproject(depths, rgbs, fx, fy, cx, cy):
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
#Class to use sam2


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
class SAM2_PC:
    def __init__(self, iou_th=0.1):
        self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else self.device
        self.iou_th = iou_th
    def predict(self, rgb_img, depth_img, bbox, scores, intrinsics, debug = False):
        """
        Gets realsense frames
        Parameters:
        - img: image to produce masks in in
        - bbox: list of bounding boxes whos masks we want

        Returns:
        - sam_mask: masks produced by sam for every bounding box
        - sam_scores: scores produced by sam for every mask
        - sam_logits: logits produced by sam for every mask
        """
        if debug:
            print(f"rgb_img.shape= {rgb_img.shape}")
            print(f"depth_img.shape= {depth_img.shape}")
        # Suppress warnings during the prediction step
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
        

        tensor_depth = torch.from_numpy(masked_depth).to(self.device)
        tensor_rgb = torch.from_numpy(masked_rgb).to(self.device)
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        points, colors = batch_backproject(tensor_depth, tensor_rgb, fx, fy, cx, cy)
        colors = colors[..., [2, 1, 0]]

        if debug:
            print(f"{points.shape=}, {colors.shape=}")

        B, N, _ = points.shape
        full_pcds = []
        full_bounding_boxes_3d = []
        full_scores = []
        # ensure CPU numpy conversion
        pts_cpu   = points.detach().cpu()
        cols_cpu  = colors.detach().cpu()
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

        #input("Press Enter to continue...")
        if debug:
            print(f"   {len(full_pcds)=}, {len(full_scores)=}, {len(full_bounding_boxes_3d)=}")

        reduced_pcds = []
        reduced_bounding_boxes_3d = []
        reduced_scores = []
        mega_array = zip(full_scores, full_pcds, full_bounding_boxes_3d)
        mega_array = sorted(mega_array, key=lambda x: x[0], reverse=True)
        while len(mega_array) > 0:
            max_score, pcd, bbox = mega_array[0]

            reduced_scores.append(max_score)
            reduced_pcds.append(pcd)
            reduced_bounding_boxes_3d.append(bbox)

            mega_array = mega_array[1:]

            if len(mega_array) == 0:
                break

            # Calculate IOU
            ious = []
            for _, _, bbox2 in mega_array:
               #print(f"{bbox.get_min_bound()=}, {bbox.get_max_bound()=}")
               #print(f"{bbox2.get_min_bound()=}, {bbox2.get_max_bound()=}")
               ious.append(iou_3d(bbox, bbox2))
               #print(f"{ious[-1]=}")

            # Filter out boxes with IOU > threshold
            mega_array = [item for item, iou in zip(mega_array, ious) if iou < self.iou_th]
        reduced_scores = torch.tensor(reduced_scores)
        reduced_scores = reduced_scores / reduced_scores.sum()
        if debug:
            print(f"   {len(reduced_pcds)=}, {len(reduced_bounding_boxes_3d)=}, {reduced_scores.shape=}")

        
        return reduced_pcds, reduced_bounding_boxes_3d, reduced_scores
    def __str__(self):
        return f"SAM2: {self.sam_predictor.model.device}"
    def __repr__(self):
        return self.__str__()
    
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
def test_OWL(left_img):
    
    owl = OWLv2()
    predicitons = owl.predict(left_img, ["phone", "water bottle"], debug=True)
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

def test_sam(rgb_img, depth_img, predictions, intrinsics, n_display = 2):
    sam = SAM2_PC()
    for querry_object, canditates in predictions.items():
        #fig, ax = plt.subplots(n_display, 2, figsize=(10* n_display, 5 * n_display))
        #print(f"{querry_object=}, {canditates['boxes'].shape=}, {canditates['scores'].shape=}")
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


if __name__=="__main__":
    print(torch.__version__, "built with CUDA", torch.version.cuda)
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
    
    
    print(f"\n\nTESTING VP")
    test_VP(cap)


    

