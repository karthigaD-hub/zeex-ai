import os
import shutil
import random

# Source folder with no-fire images
src_dir = r"D:\projects\zeex-fire-detect\src\raw\kaggle_wildfire\forest_fire\Training and Validation\nofire"

# Destination folders for YOLO dataset
dst_dirs = {
    "train": r"D:\projects\zeex-fire-detect\data\processed\fire_smoke_yolo\images\train",
    "valid": r"D:\projects\zeex-fire-detect\data\processed\fire_smoke_yolo\images\valid"
}

# Corresponding label folders
label_dirs = {k: v.replace("images", "labels") for k, v in dst_dirs.items()}

# Create label folders if they don't exist
for ldir in label_dirs.values():
    os.makedirs(ldir, exist_ok=True)

# List all images in source folder
all_images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_images)  # Shuffle to randomize train/valid split

# Split ratio
train_ratio = 0.8
train_count = int(len(all_images) * train_ratio)

# Move images and create null-class labels
for i, img_file in enumerate(all_images):
    split = "train" if i < train_count else "valid"
    src_path = os.path.join(src_dir, img_file)
    dst_path = os.path.join(dst_dirs[split], img_file)

    # Move image
    shutil.move(src_path, dst_path)

    # Create empty label for null class
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(label_dirs[split], label_file)
    open(label_path, 'w').close()

print(f"✅ Moved {len(all_images)} images and created null-class labels for train/valid split.")
