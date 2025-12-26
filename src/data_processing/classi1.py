import os
import random
import shutil

# Paths
IMAGE_DIR = "data/processed/fire_smoke_yolo/images/train"
LABEL_DIR = "data/processed/fire_smoke_yolo/labels/train"
OUT_DIR = "data/classification"

random.seed(42)

CLASSES = ["fire", "smoke", "null"]

# Create output folders
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(f"{OUT_DIR}/{split}/{cls}", exist_ok=True)

# Read all images
all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

def get_class_from_label(label_path):
    """Reads YOLO label and determines class name."""
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return "null"
    
    with open(label_path, "r") as f:
        first_line = f.readline().strip()
        if first_line == "":
            return "null"
        class_id = int(first_line.split()[0])
        return CLASSES[class_id]

# Organize by classes
data_by_class = {"fire": [], "smoke": [], "null": []}

for img in all_images:
    base = os.path.splitext(img)[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")
    cls = get_class_from_label(label_path)
    data_by_class[cls].append(img)

# Function to split list into 70/15/15
def train_val_test_split(file_list):
    random.shuffle(file_list)
    total = len(file_list)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)
    return file_list[:train_end], file_list[train_end:val_end], file_list[val_end:]

# Move files into classification dataset
for cls, files in data_by_class.items():
    train_files, val_files, test_files = train_val_test_split(files)

    for file_list, split in [(train_files, "train"), (val_files, "val"), (test_files, "test")]:
        for img in file_list:
            shutil.copy(
                os.path.join(IMAGE_DIR, img),
                f"{OUT_DIR}/{split}/{cls}/{img}"
            )

print("✅ Classification dataset created successfully!")
