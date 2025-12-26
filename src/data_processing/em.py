import os

img_dir = "data/processed/fire_smoke_yolo/images/train"
label_dir = "data/processed/fire_smoke_yolo/labels/train"

os.makedirs(label_dir, exist_ok=True)

for img in os.listdir(img_dir):
    name = img.split('.')[0] + ".txt"
    label_path = os.path.join(label_dir, name)
    
    if not os.path.exists(label_path):
        open(label_path, 'w').close()   # creates empty file

print("Empty null-class labels created.")
