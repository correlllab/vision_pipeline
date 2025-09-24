import os
import glob
import shutil
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
# ---------------------------
# Configuration helpers
# ---------------------------

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ---------------------------
# Dataset scanning
# ---------------------------

def read_data(dataset_root):
    """
    Walk the training images and pair them with matching YOLO label files (.txt).
    Returns a list of (image_path, label_path).
    NOTE: This scans dataset_root/train/images only.
    """
    image_label_pairs = []
    images_root = os.path.join(dataset_root, "train", "images")

    for root, _, files in os.walk(images_root):
        for file in files:
            if not file.lower().endswith(IMG_EXTS):
                continue

            image_path = os.path.join(root, file)

            # Compute label path by swapping 'images' -> 'labels' and extension -> .txt
            parts = os.path.normpath(image_path).split(os.sep)
            try:
                # replace the last occurrence of "images" with "labels"
                idx = len(parts) - 1 - parts[::-1].index("images")
            except ValueError:
                # no "images" segment in the path; skip
                continue

            parts[idx] = "labels"
            stem, _ = os.path.splitext(parts[-1])
            parts[-1] = stem + ".txt"
            label_path = os.sep.join(parts)

            if not (os.path.exists(image_path) and os.path.exists(label_path)):
                raise FileNotFoundError(f"Missing pair for {image_path}")

            image_label_pairs.append((image_path, label_path))

    print(f"[INFO] Found {len(image_label_pairs)} image/label pairs in {dataset_root!r}.")
    return image_label_pairs


# ---------------------------
# Split logic and label filtering
# ---------------------------

def assign_split_from_name(file_name):
    """
    Split rules:
      - If filename contains 'MixedFasteners' -> 'test' (case-insensitive)
      - Else if it has 'set_<number>' -> even -> 'train', odd -> 'val'
      - Else error (keep explicit).
    """
    if "mixedfasteners" in file_name.lower():
        return "test"

    # robustly parse ..._set_<num>_...
    chunks = file_name.replace(".", "_").split("_")
    for i, c in enumerate(chunks):
        if c.lower() == "set" and i + 1 < len(chunks):
            try:
                set_number = int(chunks[i + 1])
                if "interior" in file_name.lower():
                    quad = set_number % 4
                    if quad in (1, 2):
                        return "train"
                    elif quad in (3,):
                        return "val"
                    else:
                        return "test"
                return "train" if set_number % 4 != 0 else "val"
            except ValueError:
                pass

    raise ValueError(f"Image {file_name!r} does not contain 'set_<num>' and is not MixedFasteners.")


def write_filtered_label(src_label_path, dst_label_path, exclude_classes_by_idx, min_screen_percentage=0.0):
    """
    Copy YOLO label file while removing lines whose class index is in exclude_classes_by_idx
    OR whose bounding box normalized area (w*h) is less than min_screen_percentage.
    Keeps empty files (YOLO allows images with no objects).
    """
    os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)
    with open(src_label_path, "r", encoding="utf-8") as fin, open(dst_label_path, "w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue

            parts = line.split()

            cls_idx = int(float(parts[0]))
            w = float(parts[3])
            h = float(parts[4])
            # class exclusion
            if cls_idx in exclude_classes_by_idx:
                continue

            # area exclusion
            if w * h < min_screen_percentage:
                continue

            # keep line
            fout.write(raw if raw.endswith("\n") else raw + "\n")


# ---------------------------
# Planned dataset builder (build once)
# ---------------------------

def make_planned_dataset(data_set_path, pairs, exclude_classes=None, min_screen_percentage=0.0):
    """
    Build planned_dataset/{train,val,test}/{images,labels} once, based on filename rules.
    - Applies class exclusion by names from original data.yaml
    - Does NOT filter by camera here (camera subsets are handled by manifests)
    """
    exclude_classes = exclude_classes or []

    # Read original classes
    original_yaml_path = os.path.join(data_set_path, "data.yaml")
    with open(original_yaml_path, "r", encoding="utf-8") as f:
        original_yaml = yaml.safe_load(f)
    class_names = original_yaml["names"]
    nc = len(class_names)

    # Map exclude class names -> indices
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    exclude_classes_by_idx = set()
    for n in exclude_classes:
        if n not in name_to_idx:
            raise KeyError(f"Class name {n!r} not found in data.yaml names.")
        exclude_classes_by_idx.add(name_to_idx[n])

    # (Re)create planned structure
    planned_dataset = os.path.abspath(os.path.join(data_set_path, "planned_dataset"))  # make absolute here
    if os.path.exists(planned_dataset):
        shutil.rmtree(planned_dataset)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(planned_dataset, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(planned_dataset, split, "labels"), exist_ok=True)

    kept = {"train": 0, "val": 0, "test": 0}

    for image_path, label_path in pairs:
        file_name = os.path.basename(image_path)
        split = assign_split_from_name(file_name)

        # Copy image
        dst_img = os.path.join(planned_dataset, split, "images", file_name)
        shutil.copy2(image_path, dst_img)

        # Write filtered label
        stem, _ = os.path.splitext(file_name)
        dst_lbl = os.path.join(planned_dataset, split, "labels", stem + ".txt")
        write_filtered_label(label_path, dst_lbl, exclude_classes_by_idx, min_screen_percentage=min_screen_percentage)

        kept[split] += 1

    print(f"[STATS] kept images -> train: {kept['train']}  val: {kept['val']}  test: {kept['test']}")
    return planned_dataset, class_names


# ---------------------------
# Manifest + YAML generation (camera subsets via lists)
# ---------------------------

def list_images(root):
    """
    Return ABSOLUTE image paths. Also ensures each has a matching label file.
    """
    abs_root = os.path.abspath(root)
    candidates = glob.glob(os.path.join(abs_root, "*"))
    out = []
    dropped = 0
    for p in candidates:
        if os.path.splitext(p)[1].lower() not in IMG_EXTS:
            continue
        # derive label path by replacing 'images' with 'labels' and ext -> .txt
        parts = os.path.normpath(p).split(os.sep)
        try:
            idx = len(parts) - 1 - parts[::-1].index("images")
        except ValueError:
            continue
        parts[idx] = "labels"
        stem, _ = os.path.splitext(parts[-1])
        parts[-1] = stem + ".txt"
        lbl = os.sep.join(parts)
        if os.path.exists(lbl):
            out.append(os.path.abspath(p))
        else:
            raise ValueError(f"Image {p} has no matching label file {lbl}.")
    return sorted(out)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def write_lines(path, lines):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            # lines are already absolute; write as-is
            f.write(line + "\n")
    print(f"[WRITE] {path} ({len(lines)} lines)")

def build_manifests_and_yamls(planned_root, class_names):
    """
    For each split, write three manifest sets (all/head/hand) with ABSOLUTE image paths,
    then emit three data YAMLs that point to those .txt lists.
    """
    planned_root = os.path.abspath(planned_root)
    # 1) Collect images per split (absolute paths, label-verified)
    splits = {}
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(planned_root, split, "images")
        imgs = list_images(img_dir)  # absolute paths
        splits[split] = imgs

    # 2) Camera filters on filename only
    def cam_filter(imgs, token):
        t = token.lower()
        return [p for p in imgs if t in os.path.basename(p).lower()]

    manif_root = os.path.join(planned_root, "manifests")

    # ALL
    for split in ["train", "val", "test"]:
        write_lines(os.path.join(manif_root, "all", f"{split}.txt"), splits[split])

    # HEAD
    for split in ["train", "val", "test"]:
        head_imgs = cam_filter(splits[split], "head")
        write_lines(os.path.join(manif_root, "head", f"{split}.txt"), head_imgs)

    # HAND
    for split in ["train", "val", "test"]:
        hand_imgs = cam_filter(splits[split], "hand")
        write_lines(os.path.join(manif_root, "hand", f"{split}.txt"), hand_imgs)

    # 3) Emit YAMLs that point to the manifest files (txt paths are absolute)
    def write_yaml(name, subdir):
        data = {
            "train": os.path.abspath(os.path.join(manif_root, subdir, "train.txt")),
            "val":   os.path.abspath(os.path.join(manif_root, subdir, "val.txt")),
            "test":  os.path.abspath(os.path.join(manif_root, subdir, "test.txt")),
            "nc": len(class_names),
            "names": class_names,
        }
        out = os.path.join(planned_root, name)
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f"[WRITE] {out}")

    write_yaml("data_all.yaml",  "all")
    write_yaml("data_head.yaml", "head")
    write_yaml("data_hand.yaml", "hand")


# ---------------------------
# Main
# ---------------------------

def create_dataset(data_set_path, exclude_classes=None, min_screen_percentage=0.0):
    """
    Build planned_dataset once, then write manifests and 3 YAMLs:
      - data_all.yaml   (all cameras)
      - data_head.yaml  (only filenames containing 'head')
      - data_hand.yaml  (only filenames containing 'hand')
    """
    exclude_classes = exclude_classes or []

    # 1) Scan source train/images
    pairs = read_data(data_set_path)

    # 2) Build planned_dataset (train/val/test)
    planned_root, class_names = make_planned_dataset(
        data_set_path=data_set_path,
        pairs=pairs,
        exclude_classes=exclude_classes,
        min_screen_percentage=min_screen_percentage,
    )

    # 3) Build manifests & camera-specific YAMLs (list-of-paths style, ABSOLUTE)
    build_manifests_and_yamls(planned_root, class_names)


def vis_dataset(yaml_path, batch_size=25, splits=["train", "val", "test"]):
    yaml_path = os.path.abspath(yaml_path)
    ds_descriptor = None
    with open(yaml_path, "r", encoding="utf-8") as f:
        ds_descriptor = yaml.safe_load(f)
    assert ds_descriptor is not None
    class_names = ds_descriptor["names"]
    fig_side_length = int(np.ceil(batch_size**0.5))
    for split in splits:
    
        img_text_path = ds_descriptor[split]
        img_paths = []
        label_paths = []
        with open(img_text_path, "r", encoding="utf-8") as f:
            img_paths = [line.strip() for line in f if line.strip()]
            label_paths = [img_path.replace("images", "labels").replace(".jpg", ".txt") for img_path in img_paths]
        
        
        for i in range(0, len(img_paths), batch_size):
            fig, axes = plt.subplots(fig_side_length, fig_side_length, figsize=(30, 30))
            axes = axes.flatten()
            batch_img_paths = img_paths[i:i+batch_size]
            batch_label_paths = label_paths[i:i+batch_size]
            for img_path, label_path, ax in zip(batch_img_paths, batch_label_paths, axes):
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.axis("off")
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls_idx = int(float(parts[0]))
                        x_center = float(parts[1]) * img.shape[1]
                        y_center = float(parts[2]) * img.shape[0]
                        width = float(parts[3]) * img.shape[1]
                        height = float(parts[4]) * img.shape[0]
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        # ax.text(x_min, y_min - 10, class_names[cls_idx], color='r', fontsize=12, weight='bold')
            fig.suptitle(f"Dataset: {os.path.basename(yaml_path)}  Split: {split}  Showing {len(batch_img_paths)} samples", fontsize=16)
            plt.tight_layout()
            plt.show(block = False)
            plt.pause(0.1)  # pause to allow the figure to render


        


if __name__ == "__main__":
    # ---------------------------
    # Config
    # ---------------------------
    DATASET_ROOT = "./FastenerDataset"   # source (never modified)
    CLASSES_TO_EXCLUDE = ["Bolt", "Screw Hole", "InteriorScrew"]#[["Bolt", "Screw Hole", "InteriorScrew"], ["Bolt", "Screw Hole"], []]              # names from data.yaml (exact match), e.g. ["nut", "washer"]
    min_screen_percentage = 0#((32*32)/(1920*1080))*0.25 # e.g. 0.01 = 1% of image area
    create_dataset(DATASET_ROOT, CLASSES_TO_EXCLUDE, min_screen_percentage=min_screen_percentage)
    print(f"{min_screen_percentage=}")
    # vis_dataset("./FastenerDataset/planned_dataset/data_all.yaml", batch_size = 100)
    # plt.show()
    # vis_dataset("./FastenerDataset/planned_dataset/data_hand.yaml", batch_size = 100)
    # plt.show()
    # vis_dataset("./FastenerDataset/planned_dataset/data_head.yaml", batch_size = 100)
    # plt.show()
    vis_dataset("./FastenerDataset/planned_dataset/data_hand.yaml", batch_size = 100, splits=["test"])
    plt.show()

