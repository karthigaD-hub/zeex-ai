import os

# Paths to your images and labels
folders = {
    "train": {
        "images": "data/processed/fire_smoke_yolo/images/train",
        "labels": "data/processed/fire_smoke_yolo/labels/train"
    },
    "valid": {
        "images": "data/processed/fire_smoke_yolo/images/valid",
        "labels": "data/processed/fire_smoke_yolo/labels/valid"
    }
}

for split, paths in folders.items():
    missing_labels = []
    for img_file in os.listdir(paths["images"]):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(paths["labels"], label_file)
            if not os.path.exists(label_path):
                missing_labels.append(img_file)
    
    if missing_labels:
        print(f"[{split.upper()}] Images missing labels: {len(missing_labels)}")
        for f in missing_labels:
            print(f"  - {f}")
    else:
        print(f"[{split.upper()}] All images have labels ✅")
