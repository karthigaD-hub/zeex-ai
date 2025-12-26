import os
import shutil
from pathlib import Path

# Folders containing YOLO-ready datasets
YOLO_DATASETS = [
    "data/raw/fire_and_smoke_detection_v1_yolo8",
    "data/raw/roboflow_fire_smoke",
    "data/raw/roboflow_fire_smoke_ua3dm",
    "data/raw/kaggle_smoke_fire_yolo"
]

TARGET = Path("data/processed/fire_smoke_yolo")

def copy_dataset(dataset_path):
    dataset_path = Path(dataset_path)

    for split in ["train", "valid", "test"]:
        img_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"

        if not img_dir.exists():
            print(f"Skipping {dataset_path} — missing split: {split}")
            continue

        for img in img_dir.glob("*.*"):
            target_img = TARGET / "images" / split / img.name
            shutil.copy(img, target_img)

        for label in label_dir.glob("*.txt"):
            target_label = TARGET / "labels" / split / label.name
            shutil.copy(label, target_label)

        print(f"Merged: {dataset_path} → {split}")

def main():
    TARGET.mkdir(parents=True, exist_ok=True)

    for ds in YOLO_DATASETS:
        print(f"\nProcessing dataset: {ds}")
        copy_dataset(ds)

    print("\nDataset Merging Completed Successfully!")

if __name__ == "__main__":
    main()
