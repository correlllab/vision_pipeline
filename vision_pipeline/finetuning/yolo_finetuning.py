import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

from ultralytics import YOLOWorld, YOLO, RTDETR
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg, cfg2dict, check_cfg
from torchvision.ops import box_iou
from ultralytics.utils.ops import xywhn2xyxy
from scipy.optimize import linear_sum_assignment
import gc
import os

#TRY TO SET
ORIGINAL_DATASET = "./FastenerDataset"
# base_model = "yolov8s-worldv2.pt"
# base_model = "yolov8m-worldv2.pt"
# base_model = "yolov8l-worldv2.pt"
# base_model = "yolov8x-worldv2.pt"
# model_constructor = lambda arg: YOLOWorld(arg)


# base_model = "yolo11l-obb.pt"
# base_model = "yolov8x.pt"
# model_constructor = lambda arg: YOLO(arg)

# base_model = "rtdetr-l.pt"
# base_model = "rtdetr-x.pt"
# model_constructor = lambda arg: RTDETR(arg)

base_model = None
model_constructor = None


EPOCHS=1024
PATIENCE=100
IMG_WIDTH = int(np.ceil(1920/32)*32)  # must be multiple of 32
STARTING_LR = 1e-3
ENDING_LR = 1e-5
OPTIMIZER = "AdamW"
BATCH_SIZE = 1
DEBUG = False




base_test_cfg = {
    "task":"detect",
    "mode":"predict",
    "batch":4,
    "imgsz":IMG_WIDTH,
    "device":"cuda:0" if torch.cuda.is_available() else "cpu",
    "rect":True,
    "split": "test",
    "half":True,            # FP16 if supported; faster, same metrics
    "conf":0.25,           # doesn’t affect mAP; stable P/R readout
    "iou":0.5,              # NMS IoU (typical default)
    "max_det":300,          # plenty for fasteners
    "plots":True,           # saves PR curve, confusion matrix, etc.
    "verbose":False,
    
    # --- AUGMENTATION PARAMETERS ---
    "augment":False,
    "mixup":0.0,
}

base_train_cfg = {
    "optimizer":OPTIMIZER,
    "epochs":EPOCHS,
    "imgsz":IMG_WIDTH,
    "rect":False,
    "batch":BATCH_SIZE,
    "device":"cuda:0" if torch.cuda.is_available() else "cpu",
    "verbose":False,
    "patience":PATIENCE,
    "plots":True,
    "lr0":STARTING_LR,
    "lrf":ENDING_LR / STARTING_LR,
    "cache":False,
    "amp":True,            # Automatic Mixed Precision (AMP) training

    # --- AUGMENTATION PARAMETERS ---
    # Geometric Augmentations
    "mosaic":0.5,      # (Probability) Combine 4 images. Highly recommended.
    "mixup":0.0,       # (Probability) Mix two images and their labels.
    "scale":0.3,       # (Magnitude) Random scaling
    "multi_scale":True,
    "translate":0.05,   # (Magnitude) Random translation
    "flipud":0.5,      # (Probability) Flip up-down
    "fliplr":0.5,      # (Probability) Flip left-right
    
    # Color Augmentations
    "hsv_h":0.015,     # (Magnitude) Image HSV-Hue augmentation
    "hsv_s":0.7,       # (Magnitude) Image HSV-Saturation augmentation
    "hsv_v":0.4,       # (Magnitude) Image HSV-Value augmentation

    #loss fn
    "box":15,
    "cls":0.5,
    "dfl":1.5,
}

def visualize_and_save_iou(image_tensor, gt_boxes, pred_boxes, save_path, show=False):
    """
    Save visualization with:
      - GT in green
      - Predictions in red
    """
    img = image_tensor.detach().cpu()
    if img.dtype != torch.float32 or img.max() > 1:
        img = img.float() / 255.0
    img = img.clamp(0, 1)
    img_np = img.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_np)

    # GT boxes (green)
    if gt_boxes is not None and gt_boxes.numel() > 0:
        for (x1, y1, x2, y2) in gt_boxes.cpu().numpy():
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

    # Pred boxes (red)
    if pred_boxes is not None and pred_boxes.numel() > 0:
        for (x1, y1, x2, y2) in pred_boxes.cpu().numpy():
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    ax.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def get_average_IOU(
    test_model,
    data_yaml: str,
    project_dir: str,
    model_name: str,
    use_hungarian: bool = True,  # Use Hungarian algorithm for matching
    conf_threshold: float = 0.30,  # Lower confidence threshold
    iou_threshold: float = 0.50,    # Lower NMS threshold
):
    """
    Improved IoU computation with better handling of edge cases
    """
    data_info = check_det_dataset(data_yaml)
    img_path = data_info["test"]

    # Build eval dataloader with improved settings
    test_cfg = base_test_cfg.copy()
    test_cfg.update({
        "data": data_yaml,
        "project": project_dir,
        "name": model_name + "_iou",
        "conf": conf_threshold,
        "iou": iou_threshold,
        "max_det": 1000,  # Increased from 300
        "augment": False,  # Test Time Augmentation
    })

    cfg = get_cfg(overrides=test_cfg)
    cfg_dict = cfg2dict(cfg)
    check_cfg(cfg_dict)

    dataset = build_yolo_dataset(cfg, img_path, test_cfg['batch'], data_info)
    data_loader = build_dataloader(dataset, batch=test_cfg['batch'], workers=8)

    device = cfg.device
    test_model.to(device).eval()

    N = 0
    iou_acc = 0.0
    
    # Track statistics
    total_images = 0
    images_with_gt = 0
    images_with_pred = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(data_loader):
            images = batch["img"].to(device)
            if images.dtype != torch.float32 or images.max() > 1:
                images = images.float() / 255.0

            # Use Test Time Augmentation for better predictions
            results = test_model(
                images,
                imgsz=test_cfg["imgsz"],
                conf=test_cfg["conf"],
                iou=test_cfg["iou"],
                verbose=test_cfg["verbose"],
                augment=test_cfg["augment"],  # TTA
            )
            
            h, w = images.shape[2], images.shape[3]

            for i in range(images.shape[0]):
                total_images += 1
                
                # Get predictions with better error handling
                try:
                    pred = results[i].boxes
                    if pred is None or pred.xyxy.numel() == 0:
                        pred_boxes = torch.empty(0, 4, device=device)
                        pred_conf = torch.empty(0, device=device)
                        pred_cls = None
                    else:
                        pred_boxes = pred.xyxy
                        pred_conf = pred.conf
                        pred_cls = pred.cls if hasattr(pred, 'cls') else None
                except Exception as e:
                    if DEBUG:
                        print(f"Error getting predictions for image {i}: {e}")
                    continue

                # Get ground truth with improved error handling
                try:
                    if "batch_idx" in batch and batch["batch_idx"].numel() > 0:
                        img_mask = batch["batch_idx"] == i
                        if img_mask.sum() == 0:
                            continue  # No GT for this image
                        gt_n = batch["bboxes"][img_mask].to(device)
                    else:
                        gt_n = batch["bboxes"][i].to(device)
                        if gt_n.ndim == 1 and gt_n.numel() == 4:
                            gt_n = gt_n.unsqueeze(0)
                    
                    if gt_n.numel() == 0:
                        continue
                        
                    images_with_gt += 1
                except Exception as e:
                    if DEBUG:
                        print(f"Error getting GT for image {i}: {e}")
                    continue

                # Convert normalized coordinates with bounds checking
                try:
                    gt_boxes = xywhn2xyxy(gt_n, w=w, h=h)
                    # Ensure valid boxes
                    valid_mask = ((gt_boxes[:, 2] > gt_boxes[:, 0]) & 
                                 (gt_boxes[:, 3] > gt_boxes[:, 1]))
                    gt_boxes = gt_boxes[valid_mask]
                    
                    if gt_boxes.numel() == 0:
                        continue
                        
                    # Clamp to image bounds
                    gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clamp_(0, w)
                    gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clamp_(0, h)
                    
                except Exception as e:
                    if DEBUG:
                        print(f"Error processing GT boxes for image {i}: {e}")
                    continue

                if pred_boxes.numel() == 0:
                    N += gt_boxes.shape[0]  # Count unmatched GT boxes
                    continue
                    
                images_with_pred += 1
                os.makedirs(os.path.join(project_dir, f"{model_name}_iou_vis"), exist_ok=True)
                save_path = os.path.join(project_dir, f"{model_name}_iou_vis", f"batch{batch_idx}_img{i}.png")
                visualize_and_save_iou(images[i], gt_boxes, pred_boxes, save_path, show=DEBUG)

                
                
                # Geometric IoU (class-agnostic)
                iou_mat = box_iou(pred_boxes, gt_boxes)
                
                if use_hungarian:
                    cost = (1.0 - iou_mat).detach().cpu().numpy()
                    r, c = linear_sum_assignment(cost)

                    # clamp to valid range (paranoia)
                    r = r[r < iou_mat.shape[0]]
                    c = c[c < iou_mat.shape[1]]

                
                    pair_ious = iou_mat[r, c]
                    iou_acc += pair_ious.sum().item()
                    # denominator must count ALL GT boxes (matched or not)
                    N += len(pair_ious)
                else:
                    # no Hungarian: best IoU per GT
                    max_iou = iou_mat.max(dim=0).values
                    iou_acc += max_iou.sum().item()
                    N += gt_boxes.shape[0]

            if DEBUG and batch_idx >= 2:
                break

    avg_iou = iou_acc / N if N > 0 else 0.0
    
    if DEBUG:
        print(f"\n=== Improved IoU Results ===")
        print(f"Total images processed: {total_images}")
        print(f"Images with GT: {images_with_gt}")
        print(f"Images with predictions: {images_with_pred}")
        print(f"Total GT boxes: {N}")
        print(f"Total IoU accumulated: {iou_acc:.4f}")
        print(f"Average IoU: {avg_iou:.4f}")
    
    return avg_iou
def get_metrics(run_dir, data_yaml, project_dir, model_name):
    best_weights = os.path.join(run_dir, "weights", "best.pt")
    test_model = model_constructor(best_weights).to(base_test_cfg["device"])
    avg_iou = get_average_IOU(test_model, data_yaml, project_dir, model_name, use_hungarian=False)
    hungarian_iou = get_average_IOU(test_model, data_yaml, project_dir, model_name, use_hungarian=True)
    test_cfg = base_test_cfg.copy()
    test_cfg["data"] = data_yaml
    test_cfg["project"] = project_dir
    data_yaml.split("/")[-1].split(".")[0]
    test_cfg["name"] = model_name + "_val"
    
    test_metrics = test_model.val(**test_cfg)
    metric_dict = {
        "avg_iou": avg_iou,
        "hungarian_iou": hungarian_iou,
        "mp":   test_metrics.box.mp,       # mean precision
        "mr":   test_metrics.box.mr,       # mean recall
        "map":  test_metrics.box.map,      # mAP@0.50:0.95
        "map50":test_metrics.box.map50,    # mAP@0.50
        "map75":test_metrics.box.map75     # mAP@0.75
    }
    del test_model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # Additional cleanup

    torch.cuda.synchronize()
    gc.collect()
    # print("\n\n")
    # print(test_metrics)
    # print(f"{type(test_metrics)=}")
    # print(f"{dir(test_metrics)=}")
    # print(f"{type(test_metrics.box)=}")
    # print(f"{dir(test_metrics.box)=}")
    return metric_dict

def train_model(model, data_yaml_path, project_dir, model_name):
    # Train this fold
    train_cfg = base_train_cfg.copy()
    train_cfg["data"] = data_yaml_path
    train_cfg["project"] = project_dir
    train_cfg["name"] = model_name
    model.train(**train_cfg)
    return model

def main(experiment_str):
    project_dir = os.path.join("runs", experiment_str)

    model_name = f"PlannedModel"
    planned_model = model_constructor(base_model).to(base_train_cfg["device"])
    all_data_yaml = os.path.join(ORIGINAL_DATASET, "planned_dataset", "data_all.yaml")
    hand_data_yaml = os.path.join(ORIGINAL_DATASET, "planned_dataset", "data_hand.yaml")
    head_data_yaml = os.path.join(ORIGINAL_DATASET, "planned_dataset", "data_head.yaml")

    planned_model = train_model(planned_model, all_data_yaml, project_dir, model_name)
    del planned_model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # Additional cleanup
    torch.cuda.synchronize()  # Add this line
    gc.collect()   # Add this line
    run_dir = os.path.join(project_dir, model_name)

    all_data_metric = get_metrics(run_dir, all_data_yaml, project_dir, model_name)
    head_data_metric = get_metrics(run_dir, head_data_yaml, project_dir, model_name)
    hand_data_metric = get_metrics(run_dir, hand_data_yaml, project_dir, model_name)


    

    return hand_data_metric, head_data_metric, all_data_metric

if __name__ == "__main__":
    import SplitGeneration
    exclude_classes_sets = [["Bolt", "Screw Hole", "InteriorScrew"]]
    out_file = "./experiment_results.txt"
    with open(out_file, "w") as f:
        f.write("===== Experiments =====\n")
    min_screen_percentages = [0.0, ((32*32)/(1920*1080))*0.25]
    RTDETR_constructor = lambda arg: RTDETR(arg)
    YOLO_constructer = lambda arg: YOLO(arg)
    WORLD_constructor = lambda arg: YOLOWorld(arg)

    models_to_try = [("yolov8x-worldv2.pt", WORLD_constructor), ("rtdetr-l.pt", RTDETR_constructor), ("yolov8x.pt", YOLO_constructer), ("yolo11l-obb.pt", YOLO_constructer)]
    for model_name, constructor in models_to_try:
        base_model = model_name
        model_constructor = constructor
        for min_screen_percentage in min_screen_percentages:
            for exclude_classes in exclude_classes_sets:
                SplitGeneration.create_dataset(ORIGINAL_DATASET, exclude_classes, min_screen_percentage)
                
                experiment_results = {}
                experiment_str = f"BaseModel_{base_model}_Epochs_{EPOCHS}_BatchSize_{BATCH_SIZE}_ImgWidth_{IMG_WIDTH}_StartLR_{STARTING_LR}_EndLR_{ENDING_LR}_Optimizer_{OPTIMIZER}_ExcludeClasses_{exclude_classes}_MinScreenPerc_{min_screen_percentage}"
                experiment_str = experiment_str.replace(" ", "_").replace(",", "_").replace("[", "").replace("]", "").replace("'", "").replace('"','')
                print(f"\n\n\n\n=============Experiment {experiment_str} started.==========")
                print(f"Running {experiment_str}")
                hand_metrics, head_metrics, all_metrics = main(experiment_str=experiment_str)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # Additional cleanup
                torch.cuda.synchronize()  # Add this line
                gc.collect()   # Add this line
                print(f"{hand_metrics=}")
                print(f"{head_metrics=}")
                print(f"{all_metrics=}")

                experiment_results[experiment_str] = {"hand_metrics": hand_metrics, "head_metrics": head_metrics, "all_metrics": all_metrics}
                print(f"=============Experiment {experiment_str} completed.==========\n\n\n\n")
                with open(out_file, "a") as f:
                    f.write(f"\n\n\n\n=============Experiment {experiment_str}==========\n")
                    f.write(f"{hand_metrics=}\n")
                    f.write(f"{head_metrics=}\n")
                    f.write(f"{all_metrics=}\n")
