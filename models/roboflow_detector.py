"""
Roboflow Pothole Detection Client
Interfaces with the user's pre-trained Roboflow Serverless API model:
model_id: pothole-detection-i00zy/2
"""

import os
import io
import json
import base64
import urllib.request
import cv2
import numpy as np

ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "t1PnMzE1T5HZPqo4GJ43")
ROBOFLOW_MODEL_ID = "pothole-detection-i00zy/2"

class RoboflowPotholeDetector:
    def __init__(self, api_key=None, model_id=None, min_confidence=0.35):
        self.api_key = api_key or ROBOFLOW_API_KEY
        self.model_id = model_id or ROBOFLOW_MODEL_ID
        self.min_confidence = min_confidence

    def is_available(self):
        return bool(self.api_key and self.model_id)

    def detect(self, img_input):
        """
        Runs remote Roboflow inference.
        Returns list of detections with bounding boxes [x1, y1, x2, y2], confidence, and severity.
        """
        if isinstance(img_input, str):
            with open(img_input, "rb") as f:
                img_bytes = f.read()
            img = cv2.imread(img_input)
            h, w = img.shape[:2]
        else:
            _, buf = cv2.imencode(".jpg", img_input)
            img_bytes = buf.tobytes()
            h, w = img_input.shape[:2]

        b64_data = base64.b64encode(img_bytes).decode("ascii")
        url = f"{ROBOFLOW_API_URL}/{self.model_id}?api_key={self.api_key}"

        try:
            req = urllib.request.Request(
                url,
                data=b64_data.encode("ascii"),
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                res = json.loads(response.read().decode("utf-8"))

            raw_preds = res.get("predictions", [])
            detections = []

            for p in raw_preds:
                conf = float(p.get("confidence", 0.0))
                if conf < self.min_confidence:
                    continue

                cx = float(p.get("x", 0))
                cy = float(p.get("y", 0))
                bw = float(p.get("width", 0))
                bh = float(p.get("height", 0))

                x1 = max(0, int(cx - bw / 2.0))
                y1 = max(0, int(cy - bh / 2.0))
                x2 = min(w, int(cx + bw / 2.0))
                y2 = min(h, int(cy + bh / 2.0))

                cname = str(p.get("class", "pothole")).lower().replace(" ", "_")
                # Map to pothole if named pothole
                if "pothole" in cname:
                    standard_class = "pothole"
                    severity = "CRITICAL"
                    color = [34, 34, 220] # Red
                    desc = "Severe road cavity detected by Roboflow neural vision."
                else:
                    standard_class = "pothole"
                    severity = "CRITICAL"
                    color = [34, 34, 220]
                    desc = "Road defect detected by Roboflow."

                detections.append({
                    "class": standard_class,
                    "sub_type": "Roboflow Pothole",
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                    "severity": severity,
                    "color": color,
                    "description": desc,
                    "source": f"Roboflow Model ({self.model_id})"
                })

            return {
                "image_shape": [h, w],
                "total_hazards": len(detections),
                "model_type": "roboflow_serverless",
                "detections": detections
            }

        except Exception as e:
            print(f"[RoboflowPotholeDetector] API Error: {e}")
            return {
                "image_shape": [h, w],
                "total_hazards": 0,
                "model_type": "roboflow_error",
                "detections": []
            }

if __name__ == "__main__":
    detector = RoboflowPotholeDetector()
    print("Testing Roboflow detector availability:", detector.is_available())
