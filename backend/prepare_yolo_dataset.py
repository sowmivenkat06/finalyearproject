import os
import shutil
import random
from pathlib import Path

BASE_DIR = Path(r"D:\CANCER\backend")
DATASET_DIR = BASE_DIR / "dataset"
YOLO_DIR = BASE_DIR / "yolo_dataset"

# Create target directories
train_img_dir = YOLO_DIR / "images" / "train"
val_img_dir = YOLO_DIR / "images" / "val"
train_lbl_dir = YOLO_DIR / "labels" / "train"
val_lbl_dir = YOLO_DIR / "labels" / "val"

for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Gather all image files
image_files = []
classes = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]

for cls in classes:
    cls_folder = DATASET_DIR / cls / "CROPPED"
    if not cls_folder.exists():
        cls_folder = DATASET_DIR / cls
    
    if cls_folder.exists():
        for ext in ("*.bmp", "*.png", "*.jpg", "*.jpeg"):
            image_files.extend(list(cls_folder.glob(ext)))

print(f"Found total {len(image_files)} images.")

# Shuffle and split 80% train, 20% val
random.seed(42)
random.shuffle(image_files)

split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def process_files(file_list, img_target_dir, lbl_target_dir):
    count = 0
    for img_path in file_list:
        # Avoid filename collisions across classes by prefixing class name
        cls_name = img_path.parent.name if img_path.parent.name != "CROPPED" else img_path.parent.parent.name
        new_filename = f"{cls_name}_{img_path.name}"
        
        # Copy image
        dst_img_path = img_target_dir / new_filename
        shutil.copy2(img_path, dst_img_path)
        
        # Create default label (class 0: cell, full image box: center (0.5, 0.5), size (1.0, 1.0))
        label_filename = dst_img_path.stem + ".txt"
        dst_lbl_path = lbl_target_dir / label_filename
        with open(dst_lbl_path, "w") as f:
            f.write("0 0.5 0.5 1.0 1.0\n")
        count += 1
    return count

n_train = process_files(train_files, train_img_dir, train_lbl_dir)
n_val = process_files(val_files, val_img_dir, val_lbl_dir)

print(f"Dataset preparation complete! Train images: {n_train}, Val images: {n_val}")
