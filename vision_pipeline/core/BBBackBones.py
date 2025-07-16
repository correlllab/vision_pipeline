"""
Classes for different backbones for 2D bounding box detection.
all back bones implement a predict(img, queries, debug) method
and return a dictionary with the following structure:

{
    "query_object_1": {
        "boxes": [[x1, y1, x2, y2], ...],
        "probs": torch.tensor([prob1, prob2, ...])
    },
    "query_object_2": {
        "boxes": [[x1, y1, x2, y2], ...],
        "probs": torch.tensor([prob1, prob2, ...])
    },
    ...
    "query_object_N": {
        "boxes": [[x1, y1, x2, y2], ...],
        "probs": torch.tensor([prob1, prob2, ...])
    }
}

"""
import torch
from google import genai
from google.genai import types
import cv2
from PIL import Image

import json
import torch.nn.functional as F
from torchvision.ops import clip_boxes_to_image, remove_small_boxes, nms
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import os

import sys


from ultralytics import YOLOWorld

"""
Cheat Imports
"""
core_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(core_dir, "..")
utils_dir = os.path.join(parent_dir, "utils")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)


config_path = os.path.join(parent_dir, 'config.json')
config = json.load(open(config_path, 'r'))

fig_dir = os.path.join(parent_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(os.path.join(fig_dir, "OWLv2"), exist_ok=True)
os.makedirs(os.path.join(fig_dir, "Gemini"), exist_ok=True)

from API_KEYS import GEMINI_KEY


def display_2dCandidates(img, predicitons, window_prefix = ""):
    for query_object, prediction in predicitons.items():
        display_img = img.copy()
        for bbox, prob in zip(prediction["boxes"], prediction["probs"]):
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(display_img, f"{query_object} {prob:.4f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Display the image with bounding boxes
        cv2.imwrite(f"{fig_dir}/OWLV2/{window_prefix}{query_object}.png", display_img)
        cv2.imshow(f"{window_prefix}{query_object}", display_img)
        cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Class to use Gemini
def parse_gemini_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output
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
                thinking_budget=0#-1
            )
        )

    def get_json_response(self, img, queries, debug=True):
        text_responses = []
        errors = []
        success = False
        pil_img = Image.fromarray(img)
        queries_on_new_lines = "\n".join(queries)
        prompt = f" The objects to find are:\n{queries_on_new_lines}"
        json_response = None
        while not success and len(errors) < config["gemini_max_retries"]:
            try:
                raw_response = self.client.models.generate_content(
                    model=self.model,
                    contents = [pil_img, prompt],
                    config = self.config,
                )
                if debug:
                    print(f"GeminiResponse: {raw_response}")
                text_response = raw_response.text
                text_responses.append(text_response)
                json_text_response = parse_gemini_json(text_response)
                json_response = json.loads(json_text_response)
                success = True
                if debug:
                    print()
                    print(f"Gemini Response:")
                    print(f"   {img.shape=},  {pil_img} \n{prompt=}")
                    print(f"   {raw_response=}")
                    print(f"   {text_response=}")
                    print(f"   {json_text_response=}")
                    print(f"   {json_response=}")
                    print()

            except Exception as e:
                errors.append(e)
                print(f"Gemini try {len(errors)} Error: {e} Response: {text_response[-1]}")
                print("Retrying...")
                continue

        if not success or json_response is None:
            raise Exception(f"Gemini failed after {len(errors)} tries with errors:\n   {errors} \n and responses:\n   {text_responses}")
        for detection in json_response:
            assert "box_2d" in detection, f"Gemini response does not contain 'box_2d': {detection}"
            assert "label" in detection, f"Gemini response does not contain 'label': {detection}"
            assert len(detection["box_2d"]) == 4, f"Gemini response 'box_2d' does not contain 4 elements: {detection}"
        return json_response

    def predict(self, img, queries, debug=True):
        """
        Parameters:
        - img: image to produce bounding boxes in
        - queries: list of strings whos bounding boxes we want
        - debug: if True, prints debug information
        Returns:
        - candidates_2d: dictionary containing a list of bounding boxes
        """

        candidates_2d = {}

        json_response = self.get_json_response(img, queries, debug)
        width, height = img.shape[1], img.shape[0]

        for entry in json_response:
            normalized_bbox = entry["box_2d"]
            #print(f"{img.shape=}, {img.size=}")
            #print(f"{pil_img.size=}")
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
            if label not in candidates_2d:
                candidates_2d[label] = {"boxes": [bbox], "probs": [config["vlm_true_positive_rate"]]}
            else:
                candidates_2d[label]["boxes"].append(bbox)
                candidates_2d[label]["probs"].append(config["vlm_true_positive_rate"])
        for label in candidates_2d.keys():
            candidates_2d[label]["probs"] = torch.tensor(candidates_2d[label]["probs"], dtype=torch.float32)
        return candidates_2d


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

    def get_initial_candidates(self, img, queries, debug=False):
        #Preprocess inputs
        inputs = self.processor(text=queries, images=img, return_tensors="pt")
        inputs.to(self.device)

        #model forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([img.shape[:2]])  # (height, width)

        results = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0)[0]

        # Extract labels, boxes, and scores
        labels = results["labels"]
        boxes = results["boxes"]
        scores = results["scores"]

        boxes = clip_boxes_to_image(boxes, img.shape[:2])


        keep = remove_small_boxes(boxes, min_size=config["owl_min_2d_box_side"])
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        keep = nms(boxes, scores, iou_threshold=config["owl_iou_2d_reduction"])
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return boxes, scores, labels

    def get_probs(self, scores: torch.Tensor):
        """
        Normalizes the scores using a sigmoid function.
        Parameters:
        - scores: torch.Tensor containing the scores
        Returns:
        - probabilities: torch.Tensor containing the normalized scores
        """
        mean_score = scores.mean()
        return F.sigmoid((scores-mean_score) * config["owlv2_sigmoid_gain"])

    def predict(self, img, queries, debug = False):
        """
        Parameters:
        - img: image to produce bounding boxes in
        - queries: list of strings whos bounding boxes we want
        - debug: if True, prints debug information
        Returns:
        - out_dict: dictionary containing a list of bounding boxes and a list of scores for each query
        """
        label_lookup = {i: label for i, label in enumerate(queries)}
        boxes, scores, labels = self.get_initial_candidates(img, queries, debug)
        #get integer to text label mapping
        out_dict = {}
        #for each query, get the boxes and scores and perform NMS
        for i, label in enumerate(queries):
            text_label = label_lookup[i]

            # Filter boxes and scores for the current label
            mask = labels == i
            if mask.sum() == 0:
                continue
            instance_boxes = boxes[mask]
            instance_scores = scores[mask]


            #Get rid of low scores
            threshold = torch.quantile(instance_scores, config["owlv2_discard_percentile"])
            keep = instance_scores > threshold
            percentile_boxes  = instance_boxes[keep]
            percentile_scores = instance_scores[keep]

            # Update output dictionary
            out_dict[text_label] = {"probs": self.get_probs(percentile_scores), "boxes": percentile_boxes.tolist()}

        return out_dict

    def __str__(self):
        return f"OWLv2: {self.model.device}"
    def __repr__(self):
        return self.__str__()

class YOLO_WORLD:
    def __init__(self, checkpoint_file="YoloWorldL.pth", config_path=None, device="cuda:0"):
        """
        Initializes the YOLO World model.
        """
        self.model = YOLOWorld('yolov8x-worldv2.pt')
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)
        #print(f"YOLO World model loaded on {self.model.device}")
        #print(f"{dir(self.model)=}")

    def predict(self, img, queries, debug=False):
        """
        Parameters:
        - img: image to produce bounding boxes in
        - queries: list of strings whos bounding boxes we want
        - debug: if True, prints debug information
        Returns:
        - out_dict: dictionary containing a list of bounding boxes and a list of probabilities for each query
        """
        self.model.set_classes(queries)
        results = self.model.predict(img, show=debug)[0]
        print(f"{dir(results.boxes)=}")
        boxes = results.boxes.xyxy       # shape (N, 4)
        probs = results.boxes.conf      # shape (N,)
        cls_ids= results.boxes.cls.long()  # shape (N,)
        out_dict = {}
        for idx, query in enumerate(queries):
            # mask for detections of this class
            mask = cls_ids == idx

            # convert to Python lists / tensors
            selected_boxes  = boxes[mask].tolist()          # list of [x1,y1,x2,y2]
            selected_probs = probs[mask]                  # tensor of shape (K,)

            out_dict[query] = {
                "boxes":  selected_boxes,
                "probs": selected_probs
            }

        return out_dict
if __name__ == "__main__":
    # Test the Gemini_BB and OWLv2 classes
    img = cv2.imread("./ExampleImages/RGB_Table.jpg")
    queries = ["drill", "scissors", "soda can", "screwdriver", "wrench", "cat"]

    gemini_bb = Gemini_BB()
    gemini_results = gemini_bb.predict(img, queries, debug=True)
    display_2dCandidates(img, gemini_results, window_prefix="Gemini_")


    owl_v2 = OWLv2()
    owl_results = owl_v2.predict(img, queries, debug=True)
    display_2dCandidates(img, owl_results, window_prefix="OWLv2_")

    yolo_world = YOLO_WORLD()
    yolo_results = yolo_world.predict(img, queries, debug=True)
    display_2dCandidates(img, yolo_results, window_prefix="YOLOWorld_")