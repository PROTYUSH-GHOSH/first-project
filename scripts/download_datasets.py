"""
Road Hazard Public Dataset Integrator
Provides quick helpers and instructions to download benchmark datasets:
1. RDD2020 / RDD2022 (Global Road Damage Detection Challenge)
2. Roboflow Road Hazard & Pothole Detection Datasets
3. Kaggle Pothole & Road Cracks Dataset
"""

import os
import sys

DATASET_GUIDE = """
========================================================================
ROAD HAZARD AI - REAL WORLD BENCHMARK DATASETS
========================================================================

1. RDD2020 / RDD2022 (Road Damage Dataset)
   - Scope: 26,000+ dashcam road images captured across Japan, India, and Czechia.
   - Hazard Classes:
     * D00: Longitudinal Cracks
     * D10: Transverse Cracks
     * D20: Alligator Cracks (Crocodile cracks)
     * D40: Potholes and Manhole Defects
   - Official URL: https://github.com/sekilab/RoadDamageDetector
   - Paper: "Crowdsensing-based Road Damage Detection using Deep Learning"

2. Roboflow Universe Road Hazards
   - Scope: Pre-formatted YOLOv8 datasets for Potholes, Road Debris, and Cones.
   - Install CLI: pip install roboflow
   - Python Download snippet:
     ```python
     from roboflow import Roboflow
     rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
     project = rf.workspace("roboflow-100").project("road-damage")
     dataset = project.version(1).download("yolov8")
     ```

3. Kaggle Pothole & Road Hazard Dataset
   - Install CLI: pip install kaggle
   - Download command:
     `kaggle datasets download -d chitholian/annotated-potholes-dataset`
     `kaggle datasets download -d sriramsk/pothole-detection-dataset`

========================================================================
CURRENT LOCAL DATASET STATUS:
- Synthetic Dataset is active under: /dataset/
- Number of classes: 5 (pothole, road_crack, debris, standing_water, construction_barrier)
- Formats: YOLOv8 compatible (.jpg + .txt)
========================================================================
"""

def print_guide():
    print(DATASET_GUIDE)

if __name__ == "__main__":
    print_guide()
