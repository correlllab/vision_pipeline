import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["MPLBACKEND"] = "Agg"  # must be before importing pyplot

from ultralytics import YOLOWorld, YOLO, RTDETR
import numpy as np


import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import gc
import os

ORIGINAL_DATASET = "./FastenerDataset"

EPOCHS=1#1024*4
PATIENCE=128*2
IMG_WIDTH = int(np.ceil(1920/32)*32)  # must be multiple of 32
IMG_HEIGHT = int(np.ceil(1080/32)*32)
STARTING_LR = 1e-3
ENDING_LR = 1e-4
OPTIMIZER = "AdamW"
BATCH_SIZE = 1
DEBUG = False

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
    "cos_lr":True,
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


def train_model(model, data_yaml_path, project_dir, model_name):
    # Train this fold
    train_cfg = base_train_cfg.copy()
    train_cfg["data"] = data_yaml_path
    train_cfg["project"] = project_dir
    train_cfg["name"] = model_name
    model.train(**train_cfg)
    return model

def train_exp(model_constructor, base_model, experiment_str):
    project_dir = os.path.join("runs", experiment_str)

    model_name = f"PlannedModel"
    exp_model = model_constructor(base_model).to(base_train_cfg["device"])
    all_data_yaml = os.path.join(ORIGINAL_DATASET, "planned_dataset", "data_all.yaml")

    planned_model = train_model(exp_model, all_data_yaml, project_dir, model_name)
    del planned_model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # Additional cleanup
    torch.cuda.synchronize()  # Add this line
    gc.collect()   # Add this line
    run_dir = os.path.join(project_dir, model_name)
    return run_dir

if __name__ == "__main__":
    RTDETR_constructor = lambda arg: RTDETR(arg)
    YOLO_constructer = lambda arg: YOLO(arg)
    WORLD_constructor = lambda arg: YOLOWorld(arg)

    models_to_try = [("yolov8x-worldv2.pt", WORLD_constructor), ("rtdetr-l.pt", RTDETR_constructor), ("yolov8x.pt", YOLO_constructer), ("yolo11l.pt", YOLO_constructer)]
    
    for model_name, constructor in models_to_try:
        experiment_str = f"modelname_{model_name}_Epochs_{EPOCHS}_BatchSize_{BATCH_SIZE}_ImgWidth_{IMG_WIDTH}_StartLR_{STARTING_LR}_EndLR_{ENDING_LR}_Optimizer_{OPTIMIZER}"
        experiment_str = experiment_str.replace(" ", "_").replace(",", "_").replace("[", "").replace("]", "").replace("'", "").replace('"','')
        print(f"\n\n\n\n=============Experiment {experiment_str} started.==========")
        print(f"Running {experiment_str}")
        run_dir = train_exp(constructor, model_name, experiment_str=experiment_str)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Additional cleanup
        torch.cuda.synchronize()  # Add this line
        gc.collect()   # Add this line


