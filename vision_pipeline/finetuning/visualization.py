import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import re


# ==== Step 1. Read file ====
with open("experiment_results.txt", "r") as f:
    text = f.read()

# ==== Step 2. Regex to capture model blocks ====
# Match "==== ... ====" even with leading spaces or multiple blank lines
pattern = r"^\s*====\s*(BaseModel_.*?_Epochs.*?)\s*====\s*$([\s\S]*?)(?=^\s*====|\Z)"
matches = re.findall(pattern, text, re.MULTILINE)

results = {}
for model_name, block in matches:
    # Extract short model key: between "BaseModel_" and "_Epochs"
    short_key_match = re.search(r"BaseModel_(.*?)_Epochs", model_name)
    if short_key_match:
        short_key = short_key_match.group(1)
    else:
        short_key = model_name.replace("BaseModel_", "")

    metrics = {}
    for line in block.strip().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue

        key, val = line.split("=", 1)
        try:
            metrics[key.strip()] = ast.literal_eval(val.strip())
        except Exception as e:
            print(f"⚠️ Skipping line in {short_key}: {line} ({e})")

    results[short_key] = metrics

# # ==== Example Usage ====
# print("All models:", list(results.keys()))
# print("Hand metrics for yolov8x-worldv2.pt:", results["yolov8x-worldv2.pt"]["hand_metrics"])
# print("Hand metrics for yolov8x.pt:", results["yolov8x.pt"]["hand_metrics"])
# print("MAP50 (head) for yolo11l.pt:", results["yolo11l.pt"]["head_metrics"]["map50"])
# print("Avg IOU (all) for rtdetr-l.pt:", results["rtdetr-l.pt"]["all_metrics"]["avg_iou"])


# ==== Step 3. Collect all metric keys (use all_metrics as reference) ====
first_model = next(iter(results.values()))
metric_keys = list(first_model["all_metrics"].keys())

# ==== Step 4. Ensure ./figures exists ====
os.makedirs("figures", exist_ok=True)

# ==== Step 5. Plot grouped bar charts ====
colors = {"hand_metrics": "limegreen", "head_metrics": "salmon", "all_metrics": "skyblue"}
groups = ["hand_metrics", "head_metrics", "all_metrics"]

for metric in metric_keys:
    models = list(results.keys())
    x = np.arange(len(models))  # positions for models
    width = 0.25  # width of each bar

    plt.figure(figsize=(12, 6))

    bars = []
    for i, group in enumerate(groups):
        values = [results[m][group][metric] for m in models]
        b = plt.bar(x + (i - 1) * width, values, width, label=group.replace("_", " "), 
                    color=colors[group], edgecolor="black")
        plt.bar_label(b, fmt="%.3f", padding=3)
        bars.append(b)

    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.title(f"{metric} comparison across models")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join("figures", f"{metric}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

print("✅ Grouped figures saved in ./figures/ (hand=lime, head=salmon, all=skyblue)")