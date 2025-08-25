import os
from ultralytics import YOLOWorld
import shutil
import numpy as np

ORIGINAL_DATASET = "./FastenerDataset"
base_model = "yolov8s-worldv2.pt"
NUM_FOLDS = 5
EPOCHS=1
PATIENCE=25
IMG_WIDTH = int(np.ceil(640/32)*32)  # must be multiple of 32
STARTING_LR = 1e-3
ENDING_LR = 1e-5
OPTIMIZER = "AdamW"
BATCH_SIZE = 4

from ultralytics.utils.metrics import bbox_iou
from ultralytics.cfg import get_cfg, check_cfg, cfg2dict # <-- Import the official config builder

from ultralytics.data.utils import check_det_dataset
from ultralytics.data import build_yolo_dataset
from ultralytics.utils.ops import xywhn2xyxy
import torch
def get_average_iou(model, data_yaml_path: str, split: str = 'test', conf_threshold: float = 0.25):
    """
    Calculates the average IoU for all predicted boxes across a dataset.
    """
    # 1. Create a complete configuration object.
    print(f"\n\n\n{model=}")
    print(f"{dir(model)=}")
    print(f"{type(model.cfg)=}\n\n\n")

    cfg = model.cfg
    check_cfg(cfg2dict(cfg))
    # 2. Set up the data loader.
    data_info = check_det_dataset(data_yaml_path)
    dataset = build_yolo_dataset(
        cfg=cfg,
        batch=BATCH_SIZE, # Make sure BATCH_SIZE is defined in your script
        data=data_info,
        img_path=data_info[split],
        mode=split
    )

    all_max_ious = []
    device = model.device

    print(f"Calculating average IoU for '{split}' split...")
    # 3. Iterate through each BATCH of data.
    for batch in dataset:
        imgs = batch['img'].to(device).float() / 255.0
        
        # This call is efficient as it predicts on the whole batch at once.
        preds = model.predict(imgs, conf=conf_threshold, verbose=False)

        # 4. FIX: Add an inner loop to process each image result WITHIN the batch.
        for i, result in enumerate(preds):
            # Get predictions for the i-th image in the batch.
            pred_boxes_xyxyn = result.boxes.xyxyn.to(device)

            # Get ground truth boxes for the corresponding i-th image.
            gt_indices = batch['batch_idx'] == i
            gt_bboxes_for_image = batch['bboxes'][gt_indices]

            # Skip if there are no predictions or no ground truths for THIS image.
            if pred_boxes_xyxyn.shape[0] == 0 or gt_bboxes_for_image.shape[0] == 0:
                continue

            gt_bboxes_xyxyn = xywhn2xyxy(gt_bboxes_for_image).to(device)
            iou_matrix = bbox_iou(pred_boxes_xyxyn, gt_bboxes_xyxyn, xywh=False)

            if iou_matrix.numel() == 0:
                continue
            
            # Find the best IoU for each predicted box and add to the master list.
            max_ious_for_image, _ = torch.max(iou_matrix, dim=1)
            all_max_ious.extend(max_ious_for_image.cpu().tolist())

    # 5. Calculate the final average IoU.
    if not all_max_ious:
        print("Warning: No valid predictions found to calculate average IoU.")
        return 0.0

    average_iou = np.mean(all_max_ious)
    print(f"Finished. Found {len(all_max_ious)} predicted boxes.")
    return average_iou

def get_metrics(run_dir, data_yaml, project_dir, model_name):
    best_weights = os.path.join(run_dir, "weights", "best.pt")
    test_model = YOLOWorld(best_weights)
    average_iou = get_average_iou(test_model, data_yaml, split="test", conf_threshold=0.001)
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
        "average_iou": average_iou,
        "mp":   test_metrics.box.mp,       # mean precision
        "mr":   test_metrics.box.mr,       # mean recall
        "map":  test_metrics.box.map,      # mAP@0.50:0.95
        "map50":test_metrics.box.map50,    # mAP@0.50
        "map75":test_metrics.box.map75     # mAP@0.75
    }
    # print(test_metrics)
    print("\n\n")
    print(f"{type(test_metrics)=}")
    print(f"{dir(test_metrics)=}")
    print(f"{dir(test_metrics.box)=}")

    input("Press Enter to continue...")
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
