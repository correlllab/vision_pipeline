import os
import random
from collections import defaultdict
import shutil
import yaml



def read_data(dataset_root, camera_name=None):
    """
    Walk the training images and pair them with matching YOLO label files.
    Also compute per-image class counts.
    """
    image_label_pairs = []
    images_root = os.path.join(dataset_root, "train", "images")

    for root, dirs, files in os.walk(images_root):
        for file in files:
            lf = file.lower()
            if lf.endswith((".jpg", ".jpeg", ".png")) and (camera_name is None or camera_name in lf):
                image_path = os.path.join(root, file)
                # Resolve label path with same stem
                label_path = image_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
                for ext in [".txt"]:
                    candidate = os.path.splitext(label_path)[0] + ext
                    label_path = candidate  # YOLO labels are .txt
                assert os.path.exists(label_path) and os.path.exists(image_path), f"Missing pair for {image_path}"

                class_counts = {}
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            cls = int(float(line.split(" ")[0]))
                        except Exception:
                            # Malformed: skip counting but keep the file
                            continue
                        class_counts[cls] = class_counts.get(cls, 0) + 1

                image_label_pairs.append((image_path, label_path, class_counts))

    print(f"[INFO] Found {len(image_label_pairs)} image/label pairs in {dataset_root!r}.")
    return image_label_pairs


def get_splits(data, k, seed=None, size_penalty=0.01):
    """
    Greedy, class-balanced K-fold splitting using deviation-from-target scoring.
    data: list of tuples (image_path, label_path, {class_id:int -> count:int})
    k: number of folds (>=2)
    seed: RNG seed for determinism
    size_penalty: small penalty to discourage oversized folds (per item)
    """
    if seed is not None:
        random.seed(seed)
   
    # 1) total counts per class
    total_counts = defaultdict(int)
    for _, _, cc in data:
        for c, n in cc.items():
            total_counts[c] += n

    classes = list(total_counts.keys())

    # 2) per-fold targets
    target_per_fold = {c: total_counts[c] / float(k) for c in classes}

    # 3) order items by "rarity" (hardest first)
    def rarity_score(item):
        _, _, cc = item
        # weight rare classes higher; bigger items first to place the "hard" mass early
        return sum((cc[c] / float(total_counts[c])) for c in cc) + 1e-6 * sum(cc.values())

    items = data[:]
    random.shuffle(items)
    items.sort(key=rarity_score, reverse=True)

    # 4) tracking
    folds = [[] for _ in range(k)]
    # Use defaultdict for class counts so unseen classes default to 0 cleanly
    base_dict = {c: 0 for c in classes}
    fold_class_counts = [base_dict.copy() for _ in range(k)]

    # Seed the first min(k, len(items)) items to different folds
    seed_n = min(k, len(items))
    for i in range(seed_n):
        item = items[i]
        folds[i].append(item)
        for c, n in item[2].items():
            fold_class_counts[i][c] += n
    items = items[seed_n:]

    # 5) scoring uses DELTA deviation
    def score_fold(fold_idx, cc):
        fcc = fold_class_counts[fold_idx]
        score = 0.0
        # Only consider classes present in this item
        for c, n in cc.items():
            before = fcc[c]
            target = target_per_fold.get(c, 0.0)
            delta = abs((before + n) - target) - abs(before - target)
            # extra cost if we go beyond target
            over_after = max(0.0, (before + n) - target)
            score += delta + 0.2 * over_after
        # balance item count across folds
        score += size_penalty * len(folds[fold_idx])
        return score

    # 6) greedy assignment with tie-breaking
    for item in items:
        _, _, cc = item
        best_score = None
        best_idxs = []
        for idx in range(k):
            s = score_fold(idx, cc)
            if (best_score is None) or (s < best_score):
                best_score = s
                best_idxs = [idx]
            elif s == best_score:
                best_idxs.append(idx)
        idx = random.choice(best_idxs)
        folds[idx].append(item)
        for c, n in cc.items():
            fold_class_counts[idx][c] += n

    return folds, fold_class_counts


def move_folds(data_set_path, folds, exclude_classes):
    """
    Create ./FastenerDataset/fold_{i}/ with images/ and labels/, copy items,
    and write a YOLO data.yaml for each fold:
      - train: absolute path to generated train.txt (list of image paths)
      - val:   absolute path to previous fold's images
      - test:  absolute path to current fold's images
    """
    original_yaml_path = os.path.join(data_set_path, "data.yaml")
    with open(original_yaml_path, "r", encoding="utf-8") as f:
        original_yaml = yaml.safe_load(f)
    class_names = original_yaml["names"]
    nc = len(class_names)


    fold_dirs = []
    for i, fold in enumerate(folds):
        fold_name = f"fold_{i}"
        fold_dir = os.path.join(data_set_path, fold_name)
        fold_dirs.append(fold_dir)
        if os.path.exists(fold_dir):
            print(f"[DELETE] {fold_name}")
            shutil.rmtree(fold_dir)
        os.makedirs(os.path.join(fold_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "labels"), exist_ok=True)
        image_dir = os.path.join(fold_dir, "images")
        label_dir = os.path.join(fold_dir, "labels")
        for image_path, _, _ in fold:
            shutil.copy(image_path, image_dir)
        for _, label_path, _ in fold:
            file_name = os.path.basename(label_path)
            new_label_path = os.path.join(label_dir, file_name)
            with open(label_path, "r", encoding="utf-8") as f_old:
                with open(new_label_path, "w", encoding="utf-8") as f_new:
                    for line in f_old.readlines():
                        class_idx = int(line.split(" ")[0])
                        if class_names[class_idx] not in exclude_classes:
                            f_new.write(line)

    K = len(folds)
    
    for i, fold_dir in enumerate(fold_dirs):
        prev_i = (i - 1) % K

        # Build train list = all except current and previous
        train_dirs = [
            os.path.join(fold_dirs[j], "images")
            for j in range(K) if j not in (i, prev_i)
        ]
        val_dir = os.path.join(fold_dirs[prev_i], "images")
        test_dir = os.path.join(fold_dirs[i], "images")

        # Create train.txt with absolute image paths
        train_list = []
        for td in train_dirs:
            td_abs = os.path.abspath(td)
            for root, _, files in os.walk(td_abs):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        train_list.append(os.path.join(root, f))
        train_txt = os.path.abspath(os.path.join(fold_dir, "train.txt"))
        with open(train_txt, "w", encoding="utf-8") as tf:
            tf.write("\n".join(sorted(train_list)))

        # Absolute paths for val/test; train is the txt file
        val_dir = os.path.abspath(val_dir)
        test_dir = os.path.abspath(test_dir)

        # Build YAML
        lines = []
        lines.append(f"# Auto-generated for fold {i}")
        lines.append(f"train: {train_txt}")     # single txt path (YOLO supports list file)
        lines.append(f"val: {val_dir}")
        lines.append(f"test: {test_dir}")
        lines.append("")
        lines.append(f"nc: {nc}")
        lines.append(f"names: {class_names}")
        lines.append("")

        out_path = os.path.join(fold_dir, "data.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[WRITE] {out_path} (and {train_txt})")


def make_planned_dataset(data_set_path, data, exclude_classes, test_cam=None):

    original_yaml_path = os.path.join(data_set_path, "data.yaml")
    with open(original_yaml_path, "r", encoding="utf-8") as f:
        original_yaml = yaml.safe_load(f)
    class_names = original_yaml["names"]
    nc = len(class_names)


    planned_dataset = os.path.join(data_set_path, "planned_dataset")
    if os.path.exists(planned_dataset):
        shutil.rmtree(planned_dataset)
    os.makedirs(planned_dataset, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(planned_dataset, split)
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)

    for image_path, label_path, _ in data:
        file_name = os.path.basename(image_path)
        chunked_name = file_name.split("_")
        set_idx = chunked_name.index("set") if "set" in chunked_name else -1
        if set_idx == -1:
            raise ValueError(f"Image {file_name} does not contain 'set' in its name.")
        set_number = int(chunked_name[set_idx + 1])
        if set_number % 2 == 0:
            split = "train"
        else:
            split = "val"
        if "MixedFasteners" in file_name:
            split = "test"
        if test_cam is not None and test_cam not in image_path and split == "test":
            continue
        shutil.copy(image_path, os.path.join(planned_dataset, split, "images", file_name))
        new_label_path = os.path.join(planned_dataset, split, "labels", file_name.replace(".jpg", ".txt"))
        with open(label_path, "r", encoding="utf-8") as f_old:
                with open(new_label_path, "w", encoding="utf-8") as f_new:
                    for line in f_old.readlines():
                        class_idx = int(line.split(" ")[0])
                        if class_names[class_idx] not in exclude_classes:
                            f_new.write(line)

    

    # Create data.yaml for the final dataset
    final_yaml_path = os.path.join(planned_dataset, "data.yaml")
    with open(final_yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: {os.path.abspath(os.path.join(planned_dataset, 'train'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(planned_dataset, 'val'))}\n")
        f.write(f"test: {os.path.abspath(os.path.join(planned_dataset, 'test'))}\n")
        f.write(f"nc: {nc}\n")
        f.write(f"names: {class_names}\n")
    print(f"[WRITE] {final_yaml_path}")

        
# ---------------------------
# Main
# ---------------------------

def main(data_set_path, K, exclude_classes = [], camera_name=None, test_cam = None):
    # 1) read the dataset
    data = read_data(data_set_path, camera_name=camera_name)
    folds, fold_class_counts = get_splits(data, K, seed=42, size_penalty=0.01)

    # 2) show per-fold class totals (post-filter)
    for i, class_count in enumerate(fold_class_counts):
        print(f"[FOLD {i}] class totals: {dict(class_count)}")

    # 3) copy files into fold_* dirs under ORIGINAL_DATASET and write fold data.yaml/train.txt
    move_folds(data_set_path, folds, exclude_classes)

    # 4) create a planned dataset with train/val/test splits
    make_planned_dataset(data_set_path, data, exclude_classes, test_cam=test_cam)


if __name__ == "__main__":
    # ---------------------------
    # Config
    # ---------------------------
    data_set = "./FastenerDataset/"     # source (never modified)
    classes_to_exclude = []  # class names to remove (must match data.yaml "names")
    camera_str = None
    k = 5

    main(data_set, k, classes_to_exclude, camera_str)



