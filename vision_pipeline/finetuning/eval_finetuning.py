

import torch
import numpy as np
import os
import gc
from ultralytics import YOLOWorld, YOLO, RTDETR


from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg, cfg2dict, check_cfg
from torchvision.ops import box_iou
from ultralytics.utils.ops import xywhn2xyxy
from scipy.optimize import linear_sum_assignment

IMG_WIDTH = int(np.ceil(1920/32)*32)  # must be multiple of 32
DEBUG = True
import matplotlib
matplotlib.use("Agg")             # belt + suspenders; still before pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as patches


base_test_cfg = {
    "task":"detect",
    "mode":"predict",
    "batch":1,
    "imgsz":IMG_WIDTH,
    "device":"cuda:0" if torch.cuda.is_available() else "cpu",
    "rect":True,
    "split": "test",
    "half":True,            # FP16 if supported; faster, same metrics
    "conf":0.1,           # doesn’t affect mAP; stable P/R readout
    "iou":0.5,              # NMS IoU (typical default)
    "max_det":300,          # plenty for fasteners
    "plots":True,           # saves PR curve, confusion matrix, etc.
    "verbose":False,
    
    # --- AUGMENTATION PARAMETERS ---
    "augment":False,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "auto_augment": False,
    "erasing": 0.0,
}



def visualize_and_save_predictions(image_tensor, gt_boxes, pred_boxes, save_path, show=False):
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


def get_box_IOU(
    test_model,
    data_yaml: str,
    project_dir: str,
    model_name: str,
):
    """
    IoU computation with Hungarian matching + per-GT IoU,
    with proper handling of edge cases.
    """
    data_info = check_det_dataset(data_yaml)
    img_path = data_info["test"]

    # Build eval dataloader with improved settings
    test_cfg = base_test_cfg.copy()
    subset = os.path.splitext(os.path.basename(data_yaml))[0]
    test_cfg.update({
        "data": data_yaml,
        "project": project_dir,
        "name": model_name + "_iou" + subset,
        "conf": test_cfg["conf"],
        "iou": test_cfg["iou"],
        "nms": False,
        "max_det": 1000,
        "augment": False,
    })

    cfg = get_cfg(overrides=test_cfg)
    print(f"{cfg=}")
    cfg_dict = cfg2dict(cfg)
    check_cfg(cfg_dict)

    dataset = build_yolo_dataset(cfg, img_path, test_cfg["batch"], data_info)
    data_loader = build_dataloader(dataset, batch=test_cfg["batch"], workers=0)

    device = cfg.device
    test_model.to(device).eval()

    tp_N = 0
    tp_iou_acc = 0.0
    N = 0
    iou_acc = 0.0
    total_images = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(data_loader):
            images = batch["img"].to(device)
            if images.dtype != torch.float32 or images.max() > 1:
                images = images.float() / 255.0

            results = test_model(
                images,
                imgsz=test_cfg["imgsz"],
                conf=test_cfg["conf"],
                iou=test_cfg["iou"],
                verbose=test_cfg["verbose"],
                augment=test_cfg["augment"],
            )

            h, w = images.shape[2], images.shape[3]

            for i in range(images.shape[0]):
                total_images += 1

                # --- Predictions ---
                pred = results[i].boxes
                if pred is None or pred.xyxy.numel() == 0:
                    pred_boxes = torch.empty(0, 4, device=device)
                else:
                    pred_boxes = pred.xyxy

                # --- Ground Truth ---
                if "batch_idx" in batch:
                    img_mask = (batch["batch_idx"] == i)
                    gt_n = batch["bboxes"][img_mask].to(device)
                else:
                    gt_n = batch["bboxes"][i].to(device)
                    if gt_n.ndim == 1 and gt_n.numel() == 4:
                        gt_n = gt_n.unsqueeze(0)

                if gt_n.numel() == 0:
                    continue

                gt_boxes = xywhn2xyxy(gt_n, w=w, h=h)
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clamp_(0, w)
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clamp_(0, h)


                # --- Visualization ---
                os.makedirs(os.path.join(project_dir, f"{model_name}_iou_vis"), exist_ok=True)
                save_path = os.path.join(
                    project_dir, f"{model_name}_iou_vis", f"batch{batch_idx}_img{i}.png"
                )
                visualize_and_save_predictions(images[i], gt_boxes, pred_boxes, save_path, show=DEBUG)

                # --- IoU Calculations ---
                if pred_boxes.numel() == 0:
                    N += gt_boxes.shape[0]  # count GT even if no preds
                    continue

                iou_mat = box_iou(pred_boxes, gt_boxes)

                # Per-GT best IoU
                iou_acc += iou_mat.max(dim=0)[0].sum().item()
                N += gt_boxes.shape[0]

                # Hungarian matching for TP-only IoU
                cost = (1.0 - iou_mat).detach().cpu().numpy()
                r, c = linear_sum_assignment(cost)
                pair_ious = iou_mat[r, c]

                valid_matches_mask = pair_ious >= 0.75
                valid_pair_ious = pair_ious[valid_matches_mask]

                tp_iou_acc += valid_pair_ious.sum().item()
                tp_N += len(valid_pair_ious)

    avg_tp_iou = tp_iou_acc / tp_N if tp_N > 0 else 0.0
    avg_iou = iou_acc / N if N > 0 else 0.0

    if DEBUG:
        print(f"Total images processed: {total_images}")
        print(f"Total GT boxes: {N}")
        print(f"Total IoU accumulated: {iou_acc:.4f}")
        print(f"Average TP IoU: {avg_tp_iou:.4f}")
        print(f"Average IoU: {avg_iou:.4f}")

    return avg_tp_iou, avg_iou





def get_metrics(test_model, data_yaml, project_dir, model_name):
    tp_iou, avg_iou = get_box_IOU(test_model, data_yaml, project_dir, model_name)
    test_cfg = base_test_cfg.copy()
    test_cfg["data"] = data_yaml
    test_cfg["project"] = project_dir
    subset = data_yaml.split("/")[-1].split(".")[0]
    test_cfg["name"] = model_name + "_val" + subset
    test_metrics = test_model.val(**test_cfg)
    metric_dict = {
        "tp_iou": tp_iou,
        "avg_iou": avg_iou,
        "mp":   float(test_metrics.box.mp),       # mean precision
        "mr":   float(test_metrics.box.mr),       # mean recall
        "map":  float(test_metrics.box.map),      # mAP@0.50:0.95
        "map50":float(test_metrics.box.map50),    # mAP@0.50
        "map75":float(test_metrics.box.map75)     # mAP@0.75
    }
    
    return metric_dict



if __name__ == "__main__":
    ORIGINAL_DATASET = "./FastenerDataset"
    all_data_yaml = os.path.join(ORIGINAL_DATASET, "planned_dataset", "data_all.yaml")
    hand_data_yaml = os.path.join(ORIGINAL_DATASET, "planned_dataset", "data_hand.yaml")
    head_data_yaml = os.path.join(ORIGINAL_DATASET, "planned_dataset", "data_head.yaml")
    assert os.path.exists(all_data_yaml)
    assert os.path.exists(hand_data_yaml)
    assert os.path.exists(head_data_yaml)
    RTDETR_constructor = lambda arg: RTDETR(arg)
    YOLO_constructer = lambda arg: YOLO(arg)
    WORLD_constructor = lambda arg: YOLOWorld(arg)
    

    output_file = "./experiment_results.txt"
    with open(output_file, "a") as f:
        f.write("\n==== begin eval ====")


    for run_dir in os.listdir("runs"):
        if not os.path.isdir(os.path.join("runs", run_dir)):
            continue
        project_dir = os.path.join("runs", run_dir)
        weight_path = os.path.join(project_dir, "PlannedModel", "weights", "best.pt")
        assert os.path.exists(weight_path)
        is_world = "world" in run_dir
        is_detr = "rtdetr" in run_dir
        test_model = None
        if is_world:
            test_model = WORLD_constructor(weight_path).to(base_test_cfg["device"])
        elif is_detr:
            test_model = RTDETR_constructor(weight_path).to(base_test_cfg["device"])
        else:
            test_model = YOLO_constructer(weight_path).to(base_test_cfg["device"])
        assert test_model is not None

        hand_metrics = get_metrics(test_model, hand_data_yaml, project_dir, "PlannedModel")
        head_metrics = get_metrics(test_model, head_data_yaml, project_dir, "PlannedModel")
        all_metrics = get_metrics(test_model, all_data_yaml, project_dir, "PlannedModel")

        with open(output_file, "a") as f:
            f.write(f"\n\n\n ==== {run_dir} ====\n")
            f.write(f"{hand_metrics=}\n")
            f.write(f"{head_metrics=}\n")
            f.write(f"{all_metrics=}\n")

        del test_model
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Additional cleanup

        torch.cuda.synchronize()
        gc.collect()