import os
import shutil
import random
from pathlib import Path
random.seed(42)

base_dir = Path("data")
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

for split in ["train", "val"]:
    (images_dir / split).mkdir(parents=True, exist_ok=True)
    (labels_dir / split).mkdir(parents=True, exist_ok=True)

image_files = list(images_dir.glob("*.jpg"))
random.shuffle(image_files)

split_idx = int(len(image_files) * 0.9)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def move_files(file_list, img_dest, lbl_dest):
    for img_file in file_list:
        lbl_file = labels_dir / (img_file.stem + ".txt")
        if lbl_file.exists():
            shutil.move(str(img_file), img_dest / img_file.name)
            shutil.move(str(lbl_file), lbl_dest / lbl_file.name)

move_files(train_files, images_dir /"train", labels_dir / "train")
move_files(val_files, images_dir /"val", labels_dir / "val")

print(f"✅ Split complete: {len(train_files)} train images, {len(val_files)} val images.")
