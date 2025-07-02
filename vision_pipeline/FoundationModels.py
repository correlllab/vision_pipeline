import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from google import genai
from google.genai import types
import warnings
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import open3d as o3d

import numpy as np
import json
import random
import torch.nn.functional as F
from torchvision.ops import clip_boxes_to_image, remove_small_boxes
from transformers import OwlViTProcessor, OwlViTForObjectDetection


import os



import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)
from utils import get_points_and_colors, parse_gemini_json, my_nms
from API_KEYS import GEMINI_KEY



_script_dir = os.path.dirname(os.path.realpath(__file__))
_config_path = os.path.join(_script_dir, 'config.json')
fig_dir = os.path.join(_script_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(os.path.join(fig_dir, "SAM2"), exist_ok=True)
os.makedirs(os.path.join(fig_dir, "Gemini"), exist_ok=True)
config = json.load(open(_config_path, 'r'))



#Class to use Gemini
class Gemini_BB:
    def __init__(self):
        self.model = config["gemini_model"]
        self.client = genai.Client(api_key=GEMINI_KEY)
        system_instruction="""
Given an image and a list of objects to find, Return a set of candidates for the object as bounding boxes as a JSON array with labels.
You can identify multiple instances of the same object or no instances at all.
You should return something like:
{
  "box_2d": [x1, y1, x2, y2],
  "label": "Object1",
},
{
  "box_2d": [x1, y1, x2, y2],
  "label": "Object1"
},
...
{
  "box_2d": [x1, y1, x2, y2],
  "label": "Object1"
},
...
{
  "box_2d": [x1, y1, x2, y2],
  "label": "ObjectN"
}
Never return masks or code fencing.
        """
        safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]
        self.config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            safety_settings=safety_settings,
            response_mime_type="application/json",
            temperature=0.1,
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1
            )
        )

    def predict(self, img, queries, debug=True):
        """
        Parameters:
        - img: image to produce bounding boxes in
        - querries: list of strings whos bounding boxes we want
        - debug: if True, prints debug information
        Returns:
        - candidates_2d: dictionary containing a list of bounding boxes
        """
        queries_on_new_lines = "\n".join(queries)
        prompt = f"""
The objects to find are:
{queries_on_new_lines}
        """
        pil_img = Image.fromarray(img)
        sucess = False
        candidates_2d = {}
        text_response = []
        errors = []
        while not sucess and len(errors) < config["gemini_max_retries"]:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents = [pil_img, prompt],
                    config = self.config,
                )
                if debug:
                    print(f"{img.shape=},  {pil_img} \n{prompt=}")
                    print(f"GeminiResponse: {response}")
                text_response.append(response.text)
                response = parse_gemini_json(response.text)

                if debug:
                    print()
                    print(f"{response=}")
                    print()
                json_response = json.loads(response)
                if debug:
                    print(f"{json.dumps(json_response, indent=4)}")
                for entry in json_response:
                    normalized_bbox = entry["box_2d"]
                    #print(f"{img.shape=}, {img.size=}")
                    #print(f"{pil_img.size=}")
                    width, height = pil_img.size
                    abs_y1 = int(normalized_bbox[0]/1000 * height)
                    abs_x1 = int(normalized_bbox[1]/1000 * width)
                    abs_y2 = int(normalized_bbox[2]/1000 * height)
                    abs_x2 = int(normalized_bbox[3]/1000 * width)
                    if abs_x1 > abs_x2:
                        abs_x1, abs_x2 = abs_x2, abs_x1

                    if abs_y1 > abs_y2:
                        abs_y1, abs_y2 = abs_y2, abs_y1
                    bbox = [abs_x1, abs_y1, abs_x2, abs_y2]
                    label = entry["label"]
                    # score = entry["score"]
                    if label not in candidates_2d:
                        candidates_2d[label] = {"boxes": [bbox], "scores": [config["vlm_true_positive_rate"]]}
                        # candidates_2d[label] = {"boxes": [bbox], "scores": [score]}
                    else:
                        candidates_2d[label]["boxes"].append(bbox)
                        candidates_2d[label]["scores"].append(config["vlm_true_positive_rate"])
                        # candidates_2d[label]["scores"].append(score)
                sucess = True
            except Exception as e:
                errors.append(e)
                print(f"Gemini try {len(errors)} Error: {e} Response: {text_response[-1]}")
                print("Retrying...")
                continue
        if not sucess:
            error_str = "\n".join([str(e) for e in errors])
            raise ValueError(f"Gemini failed to produce bounding boxes after maximum retries.\n {error_str}")
        for label in candidates_2d.keys():
            candidates_2d[label]["scores"] = torch.tensor(candidates_2d[label]["scores"], dtype=torch.float32)
        return candidates_2d


def display_owl(img, predicitons, window_prefix = ""):
    for query_object, prediction in predicitons.items():
        display_img = img.copy()
        for bbox, score in zip(prediction["boxes"], prediction["scores"]):
            #bbox = prediction["box"]
            #score = prediction["score"]
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(display_img, f"{query_object} {score:.4f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Display the image with bounding boxes
        cv2.imwrite(f"{fig_dir}/OWLV2/{window_prefix}{query_object}.png", display_img)
        cv2.imshow(f"{window_prefix}{query_object}", display_img)
        cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
class OWLv2:
    def __init__(self):
        """
        Initializes the OWLv2 model and processor.
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


    def predict(self, img, querries, debug = False):
        """
        Parameters:
        - img: image to produce bounding boxes in
        - querries: list of strings whos bounding boxes we want
        - debug: if True, prints debug information
        Returns:
        - out_dict: dictionary containing a list of bounding boxes and a list of scores for each query
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
        label_lookup = {i: label for i, label in enumerate(querries)}
        all_labels = results["labels"]
        all_boxes = results["boxes"]
        all_boxes = clip_boxes_to_image(all_boxes, img.shape[:2])
        all_scores = results["scores"]
        if debug:
            temp_pred = {}
            for i, label in enumerate(querries):
                mask = all_labels == i
                text_label = label_lookup[i]
                temp_pred[text_label] = {"boxes": all_boxes[mask], "scores": all_scores[mask]}
            display_owl(img, temp_pred, window_prefix=f"All Boxes")

        keep = remove_small_boxes(all_boxes, min_size=config["owl_min_2d_box_side"])
        small_removed_boxes = all_boxes[keep]
        small_removed_scores = all_scores[keep]
        small_removed_labels = all_labels[keep]
        if debug:
            temp_pred = {}
            for i, label in enumerate(querries):
                mask = small_removed_labels == i
                text_label = label_lookup[i]
                temp_pred[text_label] = {"boxes": small_removed_boxes[mask], "scores": small_removed_scores[mask]}
            display_owl(img, temp_pred, window_prefix=f"Small Boxes Removed")

        #get integer to text label mapping
        out_dict = {}
        #for each query, get the boxes and scores and perform NMS
        for i, label in enumerate(querries):
            text_label = label_lookup[i]

            # Filter boxes and scores for the current label
            mask = small_removed_labels == i
            instance_boxes = small_removed_boxes[mask]
            instance_scores = small_removed_scores[mask]

            #Do NMS for the current label
            pruned_boxes, pruned_scores, _ = my_nms(instance_boxes.cpu(), instance_scores.cpu(), iou_threshold=config["owl_iou_2d_reduction"], three_d=False)
            pruned_boxes  = torch.stack(pruned_boxes)
            pruned_scores = torch.stack(pruned_scores)

            if debug:
                display_owl(img, {text_label: {"boxes": pruned_boxes, "scores": pruned_scores}}, window_prefix=f"Post NMS ")
            #print(f"{pruned_boxes.shape=}, {pruned_scores.shape=}")

            #Get rid of low scores
            threshold = torch.quantile(pruned_scores, config["owlv2_discard_percentile"])
            keep = pruned_scores > threshold
            filtered_boxes  = pruned_boxes[keep]
            filtered_scores = pruned_scores[keep]

            # Normalize scores
            filtered_scores = F.sigmoid(filtered_scores*config["owlv2_sigmoid_gain"])
            #mu    = filtered_scores.mean()
            #sigma = filtered_scores.std(unbiased=False)
            #z = (filtered_scores - mu) / sigma
            #filtered_scores = F.softmax(z, dim=0)
            # filtered_scores = filtered_scores / filtered_scores.sum()

            # Update output dictionary
            out_dict[text_label] = {"scores": filtered_scores, "boxes": filtered_boxes}

            if debug:
                print(f"{text_label=}")
                print(f"    {all_labels.shape=}, {all_boxes.shape=}, {all_scores.shape=}")
                print(f"    {small_removed_boxes.shape=}, {small_removed_scores.shape=}, {small_removed_labels.shape=}")
                print(f"    {instance_scores.shape=}, {instance_boxes.shape=}")
                print(f"    {pruned_boxes.shape=}, {pruned_scores.shape=}")
                print(f"    {len(keep)=}")
                print(f"    {filtered_scores.shape=}, {filtered_boxes.shape=}")
                print(f"    {threshold=}")
                print()
                display_owl(img, {text_label: {"boxes": filtered_boxes, "scores": filtered_scores}}, window_prefix=f"Top {config['owlv2_discard_percentile']} quantile ")


        if debug:
            for key in out_dict:
                print(f"{key=} {out_dict[key]['boxes'].shape=}, {out_dict[key]['scores'].shape=}")
        return out_dict

    def __str__(self):
        return f"OWLv2: {self.model.device}"
    def __repr__(self):
        return self.__str__()


#Class to use sam2
class SAM2_PC:
    def __init__(self,):
        """
        Initializes the SAM2 model and processor.
        Parameters:
        - iou_th: IoU threshold for NMS
        """
        self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else self.device
    def predict(self, rgb_img, depth_img, bbox, scores, intrinsics, debug = False, query_str=""):
        """
        Predicts 3D point clouds from RGB and depth images and bounding boxes using SAM2.
        Cleans up the point clouds and applies NMS.
        Parameters:
        - rgb_img: RGB image
        - depth_img: Depth image
        - bbox: Bounding boxes
        - intrinsics: Camera intrinsics
        - debug: If True, prints debug information
        Returns:
        - reduced_pcds: List of reduced point clouds
        - reduced_bounding_boxes_3d: List of reduced 3D bounding boxes
        - reduced_scores: List of reduced scores
        """
        #Run sam2 on all the boxes
        self.sam_predictor.set_image(rgb_img.copy())
        sam_mask = None
        sam_scores = None
        sam_logits = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            original_sam_mask, sam_scores, sam_logits = self.sam_predictor.predict(box=bbox)
        if original_sam_mask.ndim == 3:
            # single mask → add batch axis
            original_sam_mask = original_sam_mask[np.newaxis, ...]
        sam_mask = np.all(original_sam_mask, axis=1)
        if debug:
            print(f"{sam_mask.shape=}")


        #Apply mask to the depth and rgb images
        #print(f"{original_sam_mask.shape=}, {sam_mask.shape=}, {rgb_img.shape=}, {depth_img.shape=}")
        masked_depth = depth_img[None, ...] * sam_mask
        masked_rgb = rgb_img[None, ...] * sam_mask[..., None]
        #print(f"\n\n{masked_depth.shape=}, {masked_rgb.shape=}")
        #print(f"{type(masked_depth)=}, {type(masked_rgb)=}\n\n")
        if debug:
            n_fig = 2
            print(f"{masked_depth.shape=}, {masked_rgb.shape=}")
            fig, ax = plt.subplots(n_fig, 4, figsize=(20, 10))
            ax[0,0].set_title("RGB")
            ax[0,1].set_title("Depth")
            ax[0,2].set_title("Masked RGB")
            ax[0,3].set_title("Masked Depth")
            for i in range(min(n_fig, sam_mask.shape[0])):
                bbox_rgb = rgb_img.copy()
                bbox_depth = depth_img.copy()
                x_min, y_min, x_max, y_max = map(int, bbox[i])
                cv2.rectangle(bbox_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.rectangle(bbox_depth, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                ax[i, 0].imshow(bbox_rgb)
                ax[i, 1].imshow(bbox_depth)
                ax[i, 2].imshow(masked_rgb[i])
                ax[i, 3].imshow(masked_depth[i])
            fig.suptitle(f"SAM2 Masks for {query_str} {rgb_img.shape=}, {depth_img.shape=}")
            fig.tight_layout()
            fig.savefig(f"{fig_dir}/SAM2/masks_{random.randint(0,100)}.png")
            plt.show()

        #Get points and colors from masked depth and rgb images
        #print("here 1")
        tensor_depth = torch.from_numpy(masked_depth).to(self.device)
        tensor_rgb = torch.from_numpy(masked_rgb).to(self.device)
        #print("here 2")

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
        full_masked_rgb = []
        full_masked_depth = []

        pts_cpu   = points.detach().cpu()
        cols_cpu  = colors.detach().cpu()
        #for each candiate object get the point cloud unless there are too few points
        for i in range(B):
            pts = pts_cpu[i]          # (N,3)
            cls = cols_cpu[i]         # (N,3)

            # mask out void points
            # depths = pts[:, 2]
            # valid = (depths > config["min_depth"])# & (depths < config["max_depth"])
            # if debug:
            #     print(f"{valid.sum()=}")


            # if valid.sum() <  config["min_3d_points"]:
            #     continue
            # pts_valid = pts[valid]
            # cls_valid = cls[valid]

            # build Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.numpy())
            pcd.colors = o3d.utility.Vector3dVector(cls.numpy()/255)
            pcd = pcd.voxel_down_sample(voxel_size=config["voxel_size"])
            # Apply statistical outlier removal to denoise the point cloud
            if config["statistical_outlier_removal"]:
                pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=config["statistical_nb_neighbors"], std_ratio=config["statistical_std_ratio"])
            if config["radius_outlier_removal"]:
                pcd, ind = pcd.remove_radius_outlier(nb_points=config["radius_nb_points"], radius=config["radius_radius"])
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox.color = (1.0, 0.0, 0.0)  # red
            full_pcds.append(pcd)
            full_scores.append(scores[i])
            full_bounding_boxes_3d.append(bbox)
            full_masked_rgb.append(masked_rgb[i])
            full_masked_depth.append(masked_depth[i])

        full_scores = torch.tensor(full_scores)
        if debug:
            print(f"   {len(full_pcds)=}, {len(full_scores)=}, {len(full_bounding_boxes_3d)=}, {len(full_masked_rgb)=}, {len(full_masked_depth)=}")

        return full_pcds, full_bounding_boxes_3d, full_scores, full_masked_rgb, full_masked_depth
    def __str__(self):
        return f"SAM2: {self.sam_predictor.model.device}"
    def __repr__(self):
        return self.__str__()




if __name__ == "__main__":
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    profile = pipeline.start(cfg)

    sensor = profile.get_device().first_depth_sensor()
    depth_scale = sensor.get_depth_scale()

    align = rs.align(rs.stream.color)

    video_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = video_prof.get_intrinsics()

    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    rgb_img = np.asanyarray(color_frame.get_data())
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    depth_frame = aligned.get_depth_frame()
    depth_img = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth_img *= depth_scale
    cv2.imshow("RGB", rgb_img)
    cv2.imshow("Depth", depth_img)
    cv2.waitKey(0)


    I = {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.ppx,
        "cy": intrinsics.ppy,
    }

    print(f"\n\nTESTING GEMINI")
    GEM = Gemini_BB()
    predictions_2d = GEM.predict(rgb_img, ["drill", "screw driver", "wrench"], debug=True)

    print(f"\n\nTESTING SAM")
    SAM = SAM2_PC()
    for label, pred in predictions_2d.items():
        print(f"{label=}, {pred=}")
        predictions_3d = SAM.predict(rgb_img, depth_img, pred["boxes"], pred["scores"], I, debug=True, query_str=label)
