"""
Ensemble Hazard Detection Engine
Combines:
1. Roboflow Dedicated Pothole Cloud Neural Vision (pothole-detection-i00zy/2)
2. Local YOLOv8 Neural Network (trained on RDD2020 / Fine-tuned)
3. Morphological CV Localizer (offline fallback)
"""

import cv2
import numpy as np
from .cv_detector import CVHazardDetector, HAZARD_CONFIG
from .yolo_detector import YOLOHazardDetector
from .roboflow_detector import RoboflowPotholeDetector

class RoadHazardEnsemble:
    def __init__(self, yolo_weights=None, min_confidence=0.30):
        self.cv_detector = CVHazardDetector(min_confidence=min_confidence)
        self.yolo_detector = YOLOHazardDetector(model_weights_path=yolo_weights, conf_threshold=min_confidence)
        self.roboflow_detector = RoboflowPotholeDetector(min_confidence=min_confidence)

    def analyze(self, image_input, engine="auto"):
        """
        engine options:
        - 'auto': Blended Ensemble (Roboflow Potholes + YOLO Cracks & Debris)
        - 'roboflow': Pure Roboflow Pothole Model
        - 'yolo_only': Local YOLO Neural Network
        - 'cv_only': Heuristic CV Morphology
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Unable to read image at {image_input}")
        else:
            img = image_input.copy()

        h, w = img.shape[:2]
        all_detections = []
        engine_used = []

        # 1. Roboflow Cloud Model
        if engine in ["auto", "roboflow"] and self.roboflow_detector.is_available():
            rf_res = self.roboflow_detector.detect(img)
            if rf_res["detections"]:
                all_detections.extend(rf_res["detections"])
                engine_used.append("Roboflow Cloud Model (pothole-detection-i00zy)")

        # 2. Local YOLO Neural Model
        if engine in ["auto", "yolo_only"] and self.yolo_detector.is_available():
            yolo_res = self.yolo_detector.detect(img)
            if yolo_res["detections"]:
                # If in auto mode and Roboflow already found potholes, let YOLO provide cracks and debris
                if engine == "auto" and all_detections:
                    for yd in yolo_res["detections"]:
                        # Add non-potholes or potholes in different locations
                        all_detections.append(yd)
                else:
                    all_detections.extend(yolo_res["detections"])
                engine_used.append(f"YOLOv8 ({yolo_res.get('model_type', 'Neural Net')})")

        # 3. CV Heuristic Fallback
        if engine == "cv_only" or (engine == "auto" and not all_detections):
            cv_res = self.cv_detector.detect(img)
            all_detections.extend(cv_res["detections"])
            engine_used.append("Computer Vision Morphology Engine")

        # Deduplicate with NMS
        final_detections = self.cv_detector._apply_nms(all_detections, iou_thresh=0.35)
        
        # Calculate Road Risk Index (0 - 100)
        weights = {"CRITICAL": 35, "HIGH": 25, "MEDIUM": 15, "WARNING": 10}
        risk_score = 0
        for d in final_detections:
            risk_score += weights.get(d["severity"], 10) * d["confidence"]
        risk_score = min(100, int(risk_score))
        
        if risk_score > 60:
            condition = "DANGEROUS"
            condition_color = "#ef4444"
        elif risk_score > 30:
            condition = "CAUTION ADVISED"
            condition_color = "#f59e0b"
        elif risk_score > 0:
            condition = "FAIR / MONITOR"
            condition_color = "#3b82f6"
        else:
            condition = "CLEAR & SAFE"
            condition_color = "#10b981"

        return {
            "image_shape": [h, w],
            "total_hazards": len(final_detections),
            "risk_score": risk_score,
            "condition": condition,
            "condition_color": condition_color,
            "engine": " + ".join(engine_used) if engine_used else "Heuristic Engine",
            "detections": final_detections
        }

    def render_overlay(self, img_input, result):
        return self.cv_detector.render_overlay(img_input, result)
