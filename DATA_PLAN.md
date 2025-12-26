# 📄 DATA_PLAN.md  
### Fire & Smoke Detection — Dataset Sourcing & Planning  
**Project:** Zeex Fire Detect  
**Goal:** Build a reliable fire + smoke detection system using computer vision (YOLO/CNN).

---

## 1️⃣ Dataset Requirements  
We need data for two modeling approaches:

### **A. Classification (optional baseline)**
Image-level labels:  
- **fire**  
- **smoke**  
- **normal/no_fire**

Useful for: quick baseline, verifying dataset diversity.

### **B. Object Detection (main target)**
Bounding box labels:  
- **fire**  
- **smoke**

Optional advanced classes:  
- **dense_smoke**  
- **light_smoke**  
- **wildfire_fire**  
- **indoor_fire**

Formats accepted:  
- **YOLO format (.txt + .jpg)**  
- **COCO format (.json)**

---

## 2️⃣ Target Dataset Size  
Minimum recommended:

### **Classification**
| Class | Minimum Images |
|-------|----------------|
| fire | 1000 |
| smoke | 1000 |
| none | 1000 |

**Total:** 3,000+ images

### **Object Detection**
| Class | Minimum Bounding Box Samples |
|-------|------------------------------|
| fire | 800–1500 |
| smoke | 800–1500 |

**Total:** 1,600–3,000 annotated objects  
**Images needed:** 1,200–2,500

More data → better accuracy and robustness.

---

## 3️⃣ Dataset Sources (with license notes)

### **📌 Roboflow Public Datasets**
1. **Fire & Smoke Detection Dataset**  
   Link: https://public.roboflow.com  
   License: CC BY 4.0 or CC BY-SA (verify each dataset)  
   Contains annotated fire + smoke images.

2. **Wildfire Smoke Dataset**  
   License: Open for research  
   Good for outdoor scenarios.

3. **CCTV Fire/Smoke Datasets**  
   License varies — check dataset card.

---

### **📌 Kaggle**
1. **Fire Dataset**  
   Link: Search “Fire Detection Dataset” on Kaggle  
   License: Permissive (CC0 / CC BY) depending on uploader.

2. **Wildfire Smoke Dataset**  
   Contains annotated smoke.

3. **Forest Fire Images Dataset**  
   Good for environmental fire detection.

⚠ Always check each dataset's license page.

---

### **📌 Academic Research Datasets**
1. **Corsican Fire Database**  
   Source: Research papers, CC BY license  
   Useful for flame evolution.

2. **Aerial Wildfire Dataset**  
   From wildfire surveillance research.

3. **FLAME Dataset**  
   Fire segmentation dataset.

---

### **📌 Public Video Sources (for frame extraction)**
1. **YouTube Wildfire Videos (Creative Commons)**  
   Use `yt-dlp` to download frames.  
   License: Must be CC BY / reuse allowed.

2. **Surveillance Fire Footage (CC licensed)**  
   Can generate many diverse frames.

Frame extraction tool:  
```bash
ffmpeg -i video.mp4 -vf "fps=4" frames/frame_%05d.jpg
