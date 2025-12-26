import os
import cv2

# Paths to label folders
label_dirs = [
    "data/processed/fire_smoke_yolo/labels/train",
    "data/processed/fire_smoke_yolo/labels/valid"
]

# Colors for drawing
COLOR_NORMAL = (0, 255, 0)  # green for valid box
COLOR_ERROR = (0, 0, 255)   # red for out-of-bounds

# Iterate through label folders
for label_dir in label_dirs:
    image_dir = label_dir.replace("labels", "images")
    
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        
        label_path = os.path.join(label_dir, label_file)
        image_file = label_file.replace(".txt", ".jpg")
        image_path = os.path.join(image_dir, image_file)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_file}")
            continue

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Cannot open image: {image_file}")
            continue
        h, w = img.shape[:2]

        # Read labels
        with open(label_path) as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Malformed line in {label_file}: {line.strip()}")
                continue
            class_id, x, y, bw, bh = map(float, parts)
            
            # YOLO coordinates are normalized, convert to pixels
            x_center = x * w
            y_center = y * h
            box_w = bw * w
            box_h = bh * h
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            # Check if box is out of bounds
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                color = COLOR_ERROR
                print(f"Out-of-bounds box in {label_file}, line {i+1}")
            else:
                color = COLOR_NORMAL

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Show image
        cv2.imshow("Annotation QA", img)
        key = cv2.waitKey(0)  # Press any key to go to next image
        if key == 27:  # ESC to exit early
            exit()

cv2.destroyAllWindows()
print("✅ Visual annotation QA completed.")
