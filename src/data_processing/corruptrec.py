from pathlib import Path
from PIL import Image
import shutil

# -------------------------------
# Configuration: change paths
# -------------------------------
IMAGE_DIR = Path(r"D:/projects/zeex-fire-detect/data/processed/fire_smoke_yolo/images")
LABEL_DIR = Path(r"D:/projects/zeex-fire-detect/data/processed/fire_smoke_yolo/labels")
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BACKUP_DIR = Path(r"D:/projects/zeex-fire-detect/data/backup_dataset")

# -------------------------------
# Step 0: Backup dataset
# -------------------------------
BACKUP_DIR.mkdir(exist_ok=True)
for folder in [IMAGE_DIR, LABEL_DIR]:
    shutil.copytree(folder, BACKUP_DIR / folder.name, dirs_exist_ok=True)
print(f"Backup created at {BACKUP_DIR}")

# -------------------------------
# Step 1: Clean and convert images
# -------------------------------
corrupted_images = []
for img_path in IMAGE_DIR.rglob("*.*"):
    try:
        img = Image.open(img_path)
        img.verify()  # check for corruption
        # Convert unsupported formats to JPG
        if img_path.suffix.lower() not in SUPPORTED_FORMATS:
            new_path = img_path.with_suffix(".jpg")
            img = Image.open(img_path)
            img.convert("RGB").save(new_path)
            img_path.unlink()
            print(f"Converted {img_path.name} → {new_path.name}")
    except (IOError, SyntaxError) as e:
        corrupted_images.append(img_path)
        img_path.unlink()
        print(f"Removed corrupted image: {img_path.name}")

print(f"Total corrupted images removed: {len(corrupted_images)}")

# -------------------------------
# Step 2: Fix labels with out-of-bounds values
# -------------------------------
fixed_labels_count = 0
for label_path in LABEL_DIR.rglob("*.txt"):
    img_path_jpg = IMAGE_DIR / (label_path.stem + ".jpg")
    img_path_png = IMAGE_DIR / (label_path.stem + ".png")
    # Skip labels if image missing
    if not img_path_jpg.exists() and not img_path_png.exists():
        continue
    
    fixed_lines = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip invalid lines
            cls, x, y, w, h = map(float, parts)

            # Remove zero or negative size boxes
            if w <= 0 or h <= 0:
                continue

            # Fix out-of-bounds values proportionally
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            w = min(w, 1.0 - x)
            h = min(h, 1.0 - y)

            fixed_lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    
    # Overwrite fixed label
    if fixed_lines:
        with open(label_path, "w") as f:
            f.writelines(fixed_lines)
        fixed_labels_count += 1

print(f"Labels fixed: {fixed_labels_count}")

# -------------------------------
# Step 3: Remove labels with missing images
# -------------------------------
removed_labels = 0
for label_path in LABEL_DIR.rglob("*.txt"):
    img_path_jpg = IMAGE_DIR / (label_path.stem + ".jpg")
    img_path_png = IMAGE_DIR / (label_path.stem + ".png")
    if not img_path_jpg.exists() and not img_path_png.exists():
        label_path.unlink()
        removed_labels += 1

print(f"Removed {removed_labels} labels with missing images")

# -------------------------------
# Step 4: Summary
# -------------------------------
print("✅ Dataset cleaning completed!")
print(f"Corrupted images removed: {len(corrupted_images)}")
print(f"Labels fixed: {fixed_labels_count}")
print(f"Labels removed due to missing images: {removed_labels}")
print(f"Backup saved to: {BACKUP_DIR}")
print("You can now check your dataset with:\n  yolo dataset check data=your_dataset.yaml")
