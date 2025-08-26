import os
from ultralytics import YOLOWorld
import shutil
import numpy as np

ORIGINAL_DATASET = "./FastenerDataset"
base_model = "yolov8s-worldv2.pt"
NUM_FOLDS = 2
EPOCHS=512
PATIENCE=25
IMG_WIDTH = int(np.ceil(1920/32)*32)  # must be multiple of 32
STARTING_LR = 1e-3
ENDING_LR = 1e-5
OPTIMIZER = "AdamW"
BATCH_SIZE = 4

import torch
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg, cfg2dict, check_cfg
from torchvision.ops import box_iou
from ultralytics.utils.ops import xywhn2xyxy



def get_average_IOU(test_model, data_yaml, project_dir, model_name, run_dir):
    data_info = check_det_dataset(data_yaml)
    img_path = data_info["test"]
    cfg = get_cfg(overrides={
        "task":"detect",
        "mode":"predict",
        "data":data_yaml,
        "batch":BATCH_SIZE,
        "imgsz":IMG_WIDTH,
        "device":"cuda:0" if torch.cuda.is_available() else "cpu",
        "rect":True,
        "split": "test",
        "project": project_dir,
        "name": model_name +"_iou"
    })
    cfg_dict = cfg2dict(cfg)
    check_cfg(cfg_dict)
    dataset = build_yolo_dataset(cfg, img_path, BATCH_SIZE, data_info)
    data_loader = build_dataloader(dataset, batch=BATCH_SIZE, workers=8)
    # print("\n\n\n")
    N = 0
    iou_acc = 0.0
    for batch in data_loader: 
        # print(f"{type(batch)=}")
        # print(f"{dir(batch)=}")
        # print(batch.keys())
        images = batch['img'].to(cfg.device)
        images = images.float() / 255.0

        results = test_model(images)
        # print(f"{type(results)=}")
        # print(f"{dir(results)=}")
        # print(f"{len(results)=}")
        # print(f"{type(results[0])=}")
        # print(f"{dir(results[0])=}")


        for i in range(len(images)):
            pred_boxes = results[i].boxes.xyxy
            gt_boxes = batch["bboxes"][i].to(cfg.device)
            if gt_boxes.ndim == 1:
                gt_boxes = gt_boxes.unsqueeze(0)
            # print(f"{type(pred_boxes)=}")
            # # print(f"{dir(pred_boxes)=}")
            # print(f"{pred_boxes.shape=}")
            # print(f"{type(gt_boxes)=}")
            # # print(f"{dir(gt_boxes)=}")
            # print(f"{gt_boxes.shape=}")

            if gt_boxes.numel() == 0:
                continue
            if pred_boxes.numel() == 0:
                N += gt_boxes.shape[0]
                continue
            h, w = images.shape[2], images.shape[3]
            gt_boxes = xywhn2xyxy(gt_boxes, w=w, h=h)
            iou_mat = box_iou(pred_boxes, gt_boxes)
            # print(f"{type(iou_mat)=}")
            # print(f"{dir(iou_mat)=}")
            # print(f"{iou_mat.shape=}")
            # print(iou_mat)
            max_iou, _ = iou_mat.max(dim=0)
            iou_acc += max_iou.sum().item()
            N += max_iou.shape[0]
    avg_iou = iou_acc / N if N > 0 else 0.0
    return avg_iou


def get_metrics(run_dir, data_yaml, project_dir, model_name):
    best_weights = os.path.join(run_dir, "weights", "best.pt")
    test_model = YOLOWorld(best_weights)
    avg_iou = get_average_IOU(test_model, data_yaml, project_dir, model_name, run_dir)
    test_metrics = test_model.val(
            data=data_yaml,
            split="test",
            imgsz=IMG_WIDTH,      # (H, W) multiples of 32; preserves 1920x1080 scale
            rect=False,           # rectangular loader for minimal padding
            batch=BATCH_SIZE,             # auto batch
            device="cuda:0",       # auto GPU/CPU selection (same as training "cuda:0")
            half=False,            # FP16 if supported; faster, same metrics
            conf=0.001,           # doesn’t affect mAP; stable P/R readout
            iou=0.7,              # NMS IoU (typical default)
            max_det=300,          # plenty for fasteners
            plots=True,           # saves PR curve, confusion matrix, etc.
            verbose=False,
            project=project_dir,
            name=f"{model_name}_test",
        )
    metric_dict = {
        "avg_iou": avg_iou,
        "mp":   test_metrics.box.mp,       # mean precision
        "mr":   test_metrics.box.mr,       # mean recall
        "map":  test_metrics.box.map,      # mAP@0.50:0.95
        "map50":test_metrics.box.map50,    # mAP@0.50
        "map75":test_metrics.box.map75     # mAP@0.75
    }
    # print("\n\n")
    # print(test_metrics)
    # print(f"{type(test_metrics)=}")
    # print(f"{dir(test_metrics)=}")
    # print(f"{type(test_metrics.box)=}")
    # print(f"{dir(test_metrics.box)=}")
    return metric_dict

def train_model(model, epochs, data_yaml_path, project_dir, model_name):
    # Train this fold
    model.train(
        data=data_yaml_path,
        optimizer=OPTIMIZER,
        epochs=epochs,
        imgsz=IMG_WIDTH,
        rect=True,
        batch=BATCH_SIZE,
        device="cuda:0",
        project=project_dir,
        name=model_name,
        verbose=False,
        patience=PATIENCE,
        plots=True,
        lr0=STARTING_LR,
        lrf=ENDING_LR / STARTING_LR,

         # --- AUGMENTATION PARAMETERS ---
        # Geometric Augmentations
        mosaic=1.0,      # (Probability) Combine 4 images. Highly recommended.
        mixup=0.1,       # (Probability) Mix two images and their labels.
        scale=0.5,       # (Magnitude) Random scaling
        multi_scale=True,
        translate=0.1,   # (Magnitude) Random translation
        flipud=0.5,      # (Probability) Flip up-down
        fliplr=0.5,      # (Probability) Flip left-right
        
        # Color Augmentations
        hsv_h=0.015,     # (Magnitude) Image HSV-Hue augmentation
        hsv_s=0.7,       # (Magnitude) Image HSV-Saturation augmentation
        hsv_v=0.4,       # (Magnitude) Image HSV-Value augmentation
    )
    return model

def main(num_folds, epochs, experiment_str):
    fold_rows = []  # per-fold metrics
    project_dir = os.path.join("runs", experiment_str)

    
    for i in range(num_folds):
        yaml_path = os.path.join(ORIGINAL_DATASET, f"fold_{i}", "data.yaml")
        if not os.path.exists(yaml_path):
            print(f" Skipping fold {i}, no {yaml_path}")
            continue

        print(f"\n================ Starting fold {i} =================")
        print(f"Using dataset: {yaml_path}")

        model_name = f"fold{i:02d}"
        model = YOLOWorld(base_model)
        
        model = train_model(model, epochs, yaml_path, project_dir, model_name)
        run_dir = os.path.join(project_dir, model_name)

        # Safely extract metrics across versions
        metric_dict = get_metrics(run_dir, yaml_path, project_dir, model_name)
        metric_dict["fold"] = i
        fold_rows.append(metric_dict)
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
    planned_model = YOLOWorld(base_model)
    planned_data_yaml = os.path.join(ORIGINAL_DATASET, "planned_dataset", "data.yaml")
    planned_model = train_model(planned_model, epochs, planned_data_yaml, project_dir, model_name)
    run_dir = os.path.join(project_dir, model_name)

    planned_metric_dict = get_metrics(run_dir, planned_data_yaml, project_dir, model_name)

    
    return fold_summary, planned_metric_dict



if __name__ == "__main__":
    import SplitGeneration
    exclude_classes_sets = [[]]#[["Bolts", "Screw Hole"], []]
    camera_sets = [None]#[None, "head", "hand"]
    expiriment_results = {}
    shutil.rmtree("runs", ignore_errors=True)
    out_file = "./experiment_results.txt"
    with open(out_file, "w") as f:
        f.write("===== Experiments =====\n")
    for exclude_classes in exclude_classes_sets:
        for camera in camera_sets:
            for test_cam in camera_sets:
                if camera is not None and test_cam is not None and camera != test_cam:
                    continue
                experiment_str = f"exclude_classes={exclude_classes}, camera={camera}, testcam={test_cam}"
                experiment_str = experiment_str.replace(" ", "_").replace(",", "_").replace("[", "").replace("]", "").replace("'", "").replace('"','')
                print(f"\n\n\n\n=============Experiment {experiment_str} started.==========")
                print(f"Running {experiment_str}")
                SplitGeneration.main(ORIGINAL_DATASET, NUM_FOLDS, exclude_classes, camera, test_cam=test_cam)
                fold_summary, final_metrics = main(num_folds=NUM_FOLDS, epochs=EPOCHS, experiment_str=experiment_str)
                print(f"Fold Summary: {fold_summary}")
                print(f"Final Model Metrics: {final_metrics}")
                expiriment_results[experiment_str] = {"fold_summary": fold_summary, "final_metrics": final_metrics}
                print(f"=============Experiment {experiment_str} completed.==========\n\n\n\n")
                with open(out_file, "a") as f:
                    f.write(f"\n\n\n\n=============Experiment {experiment_str}==========\n")
                    f.write(f"Experiment: {experiment_str}\n")
                    f.write(f"Fold Summary: {fold_summary}\n")
                    f.write(f"Final Model Metrics: {final_metrics}\n\n")
    
    print("\n===== Experiment Results =====")
    for exp, res in expiriment_results.items():
        print("\n\n================================")
        print(f"Experiment: {exp}")
        print("Fold Summary:", res["fold_summary"])
        print("Final Metrics:", res["final_metrics"])
