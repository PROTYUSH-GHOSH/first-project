"""
YOLO Road Hazard Detector
Interfaces with Ultralytics YOLOv8 models.
Supports loading:
1. Real-World Road Damage Model (trained on RDD2020/2022 dataset: potholes, alligator cracks, etc.)
2. Custom trained weights
3. Fallback YOLOv8 base
"""

import os
from pathlib import Path
import cv2
import numpy as np

HAZARD_CONFIG = {
    "pothole": {
        "color": [34, 34, 220],       # Red in BGR
        "severity": "CRITICAL",
        "description": "Deep asphalt cavity or road depression, severe vehicle damage risk."
    },
    "road_crack": {
        "color": [0, 140, 255],       # Orange
        "severity": "MEDIUM",
        "description": "Structural fissure or alligator crack, water infiltration risk."
    },
    "debris": {
        "color": [0, 215, 255],       # Yellow/Gold
        "severity": "HIGH",
        "description": "Solid obstacle, tire tread, or loose object on driving surface."
    },
    "standing_water": {
        "color": [235, 160, 20],      # Blue/Cyan
        "severity": "MEDIUM",
        "description": "Standing water puddle or localized accumulation, aquaplaning risk."
    },
    "construction_barrier": {
        "color": [0, 69, 255],        # Orange-Red
        "severity": "WARNING",
        "description": "Work zone barricade, traffic cone, or safety drum."
    }
}

RDD_CLASS_MAPPING = {
    "alligator": "road_crack",
    "block": "road_crack",
    "crack": "road_crack",
    "longitudinal": "road_crack",
    "transverse": "road_crack",
    "edge": "road_crack",
    "pothole": "pothole"
}

class YOLOHazardDetector:
    def __init__(self, model_weights_path=None, conf_threshold=0.25):
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_type = "base"
        self.weights_path = model_weights_path
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            base_dir = Path(__file__).resolve().parent

            # 1. Check custom fine-tuned weights
            custom_weights = base_dir / "road_hazard_best.pt"
            rdd_weights = base_dir / "road_damage_yolov8.pt"

            if self.weights_path and os.path.exists(self.weights_path):
                self.model = YOLO(self.weights_path)
                self.model_type = "custom"
                print(f"[YOLOHazardDetector] Loaded specified weights: {self.weights_path}")
            elif custom_weights.exists():
                self.model = YOLO(str(custom_weights))
                self.model_type = "fine_tuned"
                print(f"[YOLOHazardDetector] Loaded custom fine-tuned weights: {custom_weights}")
            elif rdd_weights.exists():
                self.model = YOLO(str(rdd_weights))
                self.model_type = "rdd_real_world"
                print(f"[YOLOHazardDetector] Loaded Real-World RDD Road Damage model: {rdd_weights}")
            else:
                self.model = YOLO("yolov8n.pt")
                self.model_type = "yolov8_base"
                print("[YOLOHazardDetector] Loaded standard yolov8n base model.")
        except Exception as e:
            print(f"[YOLOHazardDetector] Model loading exception: {e}")
            self.model = None

    def is_available(self):
        return self.model is not None

    def detect(self, img_input):
        """
        Runs YOLO inference on image.
        Returns list of detected hazards with bounding boxes, confidence, class names, and severities.
        """
        if self.model is None:
            return {"total_hazards": 0, "detections": []}

        if isinstance(img_input, str):
            img = cv2.imread(img_input)
        else:
            img = img_input

        h, w = img.shape[:2]
        results = self.model.predict(img, conf=self.conf_threshold, verbose=False)
        
        detections = []
        raw_names = self.model.names if hasattr(self.model, "names") else {}

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                raw_label = str(raw_names.get(cls_id, f"hazard_{cls_id}")).lower()

                # Map class name based on model type
                if raw_label in RDD_CLASS_MAPPING:
                    class_name = RDD_CLASS_MAPPING[raw_label]
                    display_sub = raw_label.title()
                elif raw_label in HAZARD_CONFIG:
                    class_name = raw_label
                    display_sub = raw_label.replace("_", " ").title()
                else:
                    class_name = "debris"
                    display_sub = raw_label.title()

                meta = HAZARD_CONFIG.get(class_name, {
                    "color": [0, 255, 0],
                    "severity": "WARNING",
                    "description": "General road obstacle."
                })

                detections.append({
                    "class": class_name,
                    "sub_type": display_sub,
                    "confidence": round(conf, 2),
                    "bbox": [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
                    "severity": meta["severity"],
                    "color": meta["color"],
                    "description": meta["description"],
                    "source": "YOLO Neural Detector"
                })

        return {
            "image_shape": [h, w],
            "total_hazards": len(detections),
            "model_type": self.model_type,
            "detections": detections
        }
