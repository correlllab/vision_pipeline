import os
from ultralytics import YOLOWorld
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg, cfg2dict, check_cfg
from torchvision.ops import box_iou
from ultralytics.utils.ops import xywhn2xyxy
from scipy.optimize import linear_sum_assignment
import gc
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

#TRY TO SET
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ORIGINAL_DATASET = "./FastenerDataset"
base_model = "yolov8x-worldv2"#"yolov8s-worldv2.pt"
NUM_FOLDS = 5
EPOCHS=512
PATIENCE=50
IMG_WIDTH = int(np.ceil(1920/32)*32)  # must be multiple of 32
STARTING_LR = 5e-4
ENDING_LR = 1e-6
OPTIMIZER = "AdamW"
BATCH_SIZE = 2
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

    # --- AUGMENTATION PARAMETERS ---
    # Geometric Augmentations
    "mosaic":0.5,      # (Probability) Combine 4 images. Highly recommended.
    "mixup":0.0,       # (Probability) Mix two images and their labels.
    "scale":0.3,       # (Magnitude) Random scaling
    "multi_scale":False,
    "translate":0.05,   # (Magnitude) Random translation
    "flipud":0.5,      # (Probability) Flip up-down
    "fliplr":0.5,      # (Probability) Flip left-right
    
    # Color Augmentations
    "hsv_h":0.015,     # (Magnitude) Image HSV-Hue augmentation
    "hsv_s":0.7,       # (Magnitude) Image HSV-Saturation augmentation
    "hsv_v":0.4,       # (Magnitude) Image HSV-Value augmentation
}



def get_average_IOU(
    test_model,
    data_yaml: str,
    project_dir: str,
    model_name: str,
    class_aware: bool = False,
    use_hungarian: bool = True,
    conf_threshold: float = 0.25,  # Lower confidence threshold
    iou_threshold: float = 0.5,    # Lower NMS threshold
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
        "augment": True,  # Test Time Augmentation
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

                # Compute IoU with improved matching
                try:
                    if class_aware and "cls" in batch and pred_cls is not None:
                        # Class-aware IoU computation
                        if "batch_idx" in batch:
                            img_mask = batch["batch_idx"] == i
                            gt_cls = batch["cls"][img_mask].to(device).view(-1).long()
                        else:
                            gt_cls = batch["cls"][i].to(device).view(-1).long()
                        
                        gt_cls = gt_cls[valid_mask]  # Apply same mask as GT boxes
                        
                        for c in torch.unique(gt_cls):
                            gidx = (gt_cls == c).nonzero(as_tuple=True)[0]
                            pidx = (pred_cls.long() == c).nonzero(as_tuple=True)[0]
                            
                            if len(gidx) == 0:
                                continue
                            if len(pidx) == 0:
                                N += len(gidx)
                                continue

                            iou_mat = box_iou(pred_boxes[pidx], gt_boxes[gidx])
                            
                            if use_hungarian and len(gidx) > 0 and len(pidx) > 0:
                                try:
                                    # Pad matrix if needed for Hungarian algorithm
                                    if iou_mat.shape[0] != iou_mat.shape[1]:
                                        max_dim = max(iou_mat.shape)
                                        padded_iou = torch.zeros(max_dim, max_dim, device=device)
                                        padded_iou[:iou_mat.shape[0], :iou_mat.shape[1]] = iou_mat
                                        cost = (1.0 - padded_iou).detach().cpu().numpy()
                                    else:
                                        cost = (1.0 - iou_mat).detach().cpu().numpy()
                                        
                                    r, c_idx = linear_sum_assignment(cost)
                                    
                                    # Only use valid assignments
                                    valid_assignments = (r < iou_mat.shape[0]) & (c_idx < iou_mat.shape[1])
                                    r = r[valid_assignments]
                                    c_idx = c_idx[valid_assignments]
                                    
                                    if len(r) > 0:
                                        pair_ious = iou_mat[r, c_idx]
                                        iou_acc += pair_ious.sum().item()
                                except Exception:
                                    # Fallback to greedy matching
                                    iou_acc += iou_mat.max(dim=0).values.sum().item()
                            else:
                                iou_acc += iou_mat.max(dim=0).values.sum().item()
                            
                            N += len(gidx)
                    else:
                        # Geometric IoU (class-agnostic)
                        iou_mat = box_iou(pred_boxes, gt_boxes)
                        
                        if use_hungarian and iou_mat.shape[0] > 0 and iou_mat.shape[1] > 0:
                            try:
                                cost = (1.0 - iou_mat).detach().cpu().numpy()
                                r, c_idx = linear_sum_assignment(cost)
                                
                                # Ensure we don't exceed matrix dimensions
                                r = r[r < iou_mat.shape[0]]
                                c_idx = c_idx[c_idx < iou_mat.shape[1]]
                                
                                if len(r) > 0 and len(c_idx) > 0:
                                    pair_ious = iou_mat[r, c_idx]
                                    iou_acc += pair_ious.sum().item()
                                    N += len(c_idx)  # Count matched GT boxes
                                else:
                                    N += gt_boxes.shape[0]  # Count all GT boxes as unmatched
                            except Exception:
                                # Fallback to greedy best match per GT
                                max_iou = iou_mat.max(dim=0).values
                                iou_acc += max_iou.sum().item()
                                N += gt_boxes.shape[0]
                        else:
                            max_iou = iou_mat.max(dim=0).values
                            iou_acc += max_iou.sum().item()
                            N += gt_boxes.shape[0]
                            
                except Exception as e:
                    if DEBUG:
                        print(f"Error computing IoU for image {i}: {e}")
                    N += gt_boxes.shape[0]  # Count GT boxes as unmatched
                    continue

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
    test_model = YOLOWorld(best_weights).to(base_test_cfg["device"])
    avg_iou = get_average_IOU(test_model, data_yaml, project_dir, model_name)

    test_cfg = base_test_cfg.copy()
    test_cfg["data"] = data_yaml
    test_cfg["project"] = project_dir
    test_cfg["name"] = model_name + "_val"
    
    test_metrics = test_model.val(**test_cfg)
    metric_dict = {
        "avg_iou": avg_iou,
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

def main(num_folds, experiment_str, do_folds = False):
    fold_rows = []  # per-fold metrics
    project_dir = os.path.join("runs", experiment_str)

    if do_folds:
        for i in range(num_folds):
            yaml_path = os.path.join(ORIGINAL_DATASET, f"fold_{i}", "data.yaml")
            if not os.path.exists(yaml_path):
                print(f" Skipping fold {i}, no {yaml_path}")
                continue

            print(f"\n================ Starting fold {i} =================")
            print(f"Using dataset: {yaml_path}")

            model_name = f"fold{i:02d}"
            model = YOLOWorld(base_model).to(base_train_cfg["device"])
            
            model = train_model(model, yaml_path, project_dir, model_name)
            run_dir = os.path.join(project_dir, model_name)

            # Safely extract metrics across versions
            metric_dict = get_metrics(run_dir, yaml_path, project_dir, model_name)
            metric_dict["fold"] = i
            fold_rows.append(metric_dict)
            print(f"Fold {i} metrics: {metric_dict}")
            del model
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Additional cleanup
            torch.cuda.synchronize()  # Add this line
            gc.collect()   # Add this line
        
        # Compute averages (ignore None)
        def _avg(key):
            vals = [r[key] for r in fold_rows]
            return sum(vals) / len(vals) if vals else None

        fold_summary = {
            "folds_count": len(fold_rows),
            "iou": _avg("avg_iou"),
            "avg_mp":   _avg("mp"),
            "avg_mr":   _avg("mr"),
            "avg_map":  _avg("map"),
            "avg_map50":_avg("map50"),
            "avg_map75":_avg("map75"),
        }
        print("\n===== K-Fold Summary =====")
        for k, v in fold_summary.items():
            print(f"{k}: {v}")
    

    model_name = f"PlannedModel"
    planned_model = YOLOWorld(base_model).to(base_train_cfg["device"])
    planned_data_yaml = os.path.join(ORIGINAL_DATASET, "planned_dataset", "data.yaml")
    planned_model = train_model(planned_model, planned_data_yaml, project_dir, model_name)
    run_dir = os.path.join(project_dir, model_name)

    planned_metric_dict = get_metrics(run_dir, planned_data_yaml, project_dir, model_name)
    del planned_model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # Additional cleanup
    torch.cuda.synchronize()  # Add this line
    gc.collect()   # Add this line

    return fold_summary, planned_metric_dict



if __name__ == "__main__":
    import SplitGeneration
    exclude_classes_sets = [["Bolts", "Screw Hole"], []]#[[]]
    camera_sets = [None, "head", "hand"] #[None]
    expiriment_results = {}
    shutil.rmtree("runs", ignore_errors=True)
    out_file = "./experiment_results.txt"
    with open(out_file, "w") as f:
        f.write("===== Experiments =====\n")
    for exclude_classes in exclude_classes_sets:
        for camera in camera_sets:
            for test_cam in ["head", "hand"]:
                if camera is not None and camera == test_cam:
                    continue
                experiment_str = f"exclude_classes={exclude_classes}, camera={camera}, testcam={test_cam}"
                experiment_str = experiment_str.replace(" ", "_").replace(",", "_").replace("[", "").replace("]", "").replace("'", "").replace('"','')
                print(f"\n\n\n\n=============Experiment {experiment_str} started.==========")
                print(f"Running {experiment_str}")
                SplitGeneration.main(ORIGINAL_DATASET, NUM_FOLDS, exclude_classes, camera, test_cam=test_cam)
                fold_summary, final_metrics = main(num_folds=NUM_FOLDS, experiment_str=experiment_str)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # Additional cleanup
                torch.cuda.synchronize()  # Add this line
                gc.collect()   # Add this line
                print(f"Fold Summary: {fold_summary}")
                print(f"Final Model Metrics: {final_metrics}")
                for k, v in fold_summary.items():
                    print(f"{k}: {type(v)}")
                for k, v in final_metrics.items():
                    print(f"{k}: {type(v)}")
                expiriment_results[experiment_str] = {"fold_summary": fold_summary, "final_metrics": final_metrics}
                print(f"=============Experiment {experiment_str} completed.==========\n\n\n\n")
                with open(out_file, "a") as f:
                    f.write(f"\n\n\n\n=============Experiment {experiment_str}==========\n")
                    f.write(f"Experiment: {experiment_str}\n")
                    f.write(f"Fold Summary: {fold_summary}\n")
                    f.write(f"Final Model Metrics: {final_metrics}\n\n")
    
    
    # Sort results by final model's average IoU in descending order
    sorted_results = sorted(
        expiriment_results.items(),
        key=lambda item: item[1]['final_metrics'].get('avg_iou', 0.0),
        reverse=True
    )

    print("\n===== Experiment Results (Sorted by Final Model IoU) =====")
    for exp, res in sorted_results:
        print("\n\n================================")
        print(f"Experiment: {exp}")
        print("Fold Summary:", res["fold_summary"])
        print("Final Metrics:", res["final_metrics"])
