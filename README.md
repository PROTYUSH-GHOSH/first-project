# RoadGuard AI — Multi-Hazard Computer Vision Model & Safety System

An AI Computer Vision & Deep Learning system designed to identify, localize, and classify road hazards in dashcam footage, surveillance feeds, and road survey imagery.

---

## 🎯 Hazard Categories & Severity Matrix

| Hazard Class | Visual Signature | Severity | Automated Infrastructure Recommendation |
| :--- | :--- | :--- | :--- |
| **`pothole`** | Depressed asphalt cavity, dark inner hollow, crumbled edge rubble | **CRITICAL** | Urgent highway patching; cold/hot mix asphalt dispatch |
| **`road_crack`** | Longitudinal, transverse, or alligator fracture network | **MEDIUM** | Bitumen crack sealing to prevent base water infiltration |
| **`debris`** | Solid lost cargo, shredded tire treads, rocks, fallen obstacles | **HIGH** | Lane obstacle alert; rapid highway clearance crew |
| **`standing_water`** | Specular reflective puddle, aquaplaning surface accumulation | **MEDIUM** | Aquaplaning advisory; roadside storm drain unblocking |
| **`construction_barrier`** | High-visibility orange/white safety cones, work drums, barricades | **WARNING** | Work-zone speed reduction advisory (30 km/h) |

---

## 🏗️ Architecture & Engines

1. **Ensemble Detection Engine (`models/ensemble.py`)**:
   - Integrates Deep Learning (YOLOv8) with Morphological Computer Vision.
   - Computes a composite **Road Risk Index (0-100)**:
     $$\text{Risk Score} = \min\left(100, \sum_{i} w_{i} \times \text{Confidence}_{i}\right)$$
   - Determines real-time road condition: `SAFE`, `MONITOR`, `CAUTION ADVISED`, or `DANGEROUS`.

2. **Morphological & Texture Computer Vision Engine (`models/cv_detector.py`)**:
   - Zero-latency, 100% offline edge inference.
   - Evaluates road geometry, asphalt luminance contrast, HSV color space (cones/barriers), specular reflectance (puddles), and Canny edge branching (cracks).

3. **YOLO Deep Learning Engine (`models/yolo_detector.py` & `scripts/train_yolo.py`)**:
   - Built on Ultralytics YOLOv8 with support for custom weights.
   - Full training pipeline with dataset validation and best weight checkpointing.

4. **Synthetic Hazard Generator (`scripts/generate_synthetic_data.py`)**:
   - Programmatically synthesizes realistic road scenes with perspective vanishing points, asphalt texturing, lane markings, and embedded hazards.
   - Generates YOLO format ground-truth labels (`.txt`).

5. **Interactive Web Dashboard & REST API (`web/app.py`)**:
   - Real-time hazard bounding box visualization with confidence & severity badges.
   - Drag-and-drop custom image uploader.
   - Quick one-click scenario testing carousel.
   - Live dashcam simulation player.
   - Automated work order generation.

---

## 🚀 Quick Start Guide

### 1. Launch the Web Application
```bash
python "C:\Users\PROTYUSH GHOSH\.gemini\antigravity\scratch\road_hazard_ai\web\app.py"
```
Then open your browser at **`http://127.0.0.1:5000`**.

### 2. Generate More Synthetic Training Samples
```bash
python "C:\Users\PROTYUSH GHOSH\.gemini\antigravity\scratch\road_hazard_ai\scripts\generate_synthetic_data.py"
```

### 3. Fine-Tune the YOLO Model
```bash
python "C:\Users\PROTYUSH GHOSH\.gemini\antigravity\scratch\road_hazard_ai\scripts\train_yolo.py" --epochs 20 --batch 8 --device cpu
```

### 4. Integrate Public Real-World Datasets
Run:
```bash
python "C:\Users\PROTYUSH GHOSH\.gemini\antigravity\scratch\road_hazard_ai\scripts\download_datasets.py"
```
Supports **RDD2020/RDD2022** (Road Damage Dataset) and **Roboflow Universe** road hazard datasets.
