import os

# Paths
img_dir = "data/processed/fire_smoke_yolo/images/train"   # folder with null images
label_dir = "data/processed/fire_smoke_yolo/labels/train"  # YOLO labels folder

os.makedirs(label_dir, exist_ok=True)

# Parameters for null class label
class_index = 2        # index of null class
center_x = 0.5         # normalized center x
center_y = 0.5         # normalized center y
width = 0.01           # normalized width
height = 0.01          # normalized height

for img_file in os.listdir(img_dir):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    txt_file = os.path.splitext(img_file)[0] + ".txt"
    txt_path = os.path.join(label_dir, txt_file)

    # Skip if label already exists
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
        continue

    # Create label for null class
    with open(txt_path, 'w') as f:
        f.write(f"{class_index} {center_x} {center_y} {width} {height}\n")

print("Null-class labels generated for all images without labels.")
