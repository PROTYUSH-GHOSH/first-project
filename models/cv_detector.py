"""
Computer Vision Road Hazard Detector
Analyzes road imagery to detect and classify road hazards using texture, morphology,
color segmentation, and spatial road geometry.
Supported hazard classes:
- pothole
- road_crack
- debris
- standing_water
- construction_barrier
"""

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

class CVHazardDetector:
    def __init__(self, min_confidence=0.45):
        self.min_confidence = min_confidence

    def _get_road_roi(self, img):
        """Estimates the drivable road surface mask assuming perspective geometry."""
        h, w = img.shape[:2]
        horizon_y = int(h * 0.35)
        
        # Trapezoid drivable zone
        pts = np.array([
            [int(w * 0.25), horizon_y],
            [int(w * 0.75), horizon_y],
            [int(w * 0.98), h],
            [int(w * 0.02), h]
        ], dtype=np.int32)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask, horizon_y

    def detect(self, img_input):
        """
        Runs hazard detection on an image (numpy array or filepath).
        Returns a dictionary of detections with bounding boxes, labels, confidence, and severity.
        """
        if isinstance(img_input, str):
            img = cv2.imread(img_input)
            if img is None:
                raise ValueError(f"Could not read image from {img_input}")
        else:
            img = img_input.copy()

        h, w = img.shape[:2]
        road_mask, horizon_y = self._get_road_roi(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        detections = []

        # 1. Detect Construction Barriers (Orange / bright reflective hazard cones)
        # Orange in HSV: H: 5-25, S: 130-255, V: 120-255
        lower_orange = np.array([5, 120, 120])
        upper_orange = np.array([24, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        orange_mask = cv2.bitwise_and(orange_mask, road_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        orange_clean = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(orange_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            area = cv2.contourArea(c)
            if area > 180:
                x, y, bw, bh = cv2.boundingRect(c)
                aspect_ratio = bh / float(bw)
                if aspect_ratio >= 0.8: # Tall or upright object
                    confidence = min(0.95, 0.65 + (area / (w * h)) * 10)
                    detections.append({
                        "class": "construction_barrier",
                        "confidence": round(float(confidence), 2),
                        "bbox": [x, y, x + bw, y + bh],
                        "severity": HAZARD_CONFIG["construction_barrier"]["severity"],
                        "color": HAZARD_CONFIG["construction_barrier"]["color"],
                        "description": HAZARD_CONFIG["construction_barrier"]["description"]
                    })

        # 2. Detect Standing Water (Specular reflections / high brightness puddles surrounded by asphalt)
        # Inside the road zone, specular reflection has low saturation and high value
        lower_water = np.array([0, 0, 175])
        upper_water = np.array([180, 55, 255])
        specular_mask = cv2.inRange(hsv, lower_water, upper_water)
        specular_mask = cv2.bitwise_and(specular_mask, road_mask)
        # Filter out the sky / horizon
        specular_mask[:horizon_y + 15, :] = 0
        
        # Puddles are generally elliptical and horizontal
        kernel_water = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
        water_clean = cv2.morphologyEx(specular_mask, cv2.MORPH_CLOSE, kernel_water)
        contours, _ = cv2.findContours(water_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            area = cv2.contourArea(c)
            if 300 < area < (w * h * 0.15):
                x, y, bw, bh = cv2.boundingRect(c)
                aspect_ratio = bw / float(bh)
                # Puddles spread horizontally
                if aspect_ratio > 1.1:
                    confidence = min(0.92, 0.60 + (area / 1000.0) * 0.05)
                    detections.append({
                        "class": "standing_water",
                        "confidence": round(float(confidence), 2),
                        "bbox": [x, y, x + bw, y + bh],
                        "severity": HAZARD_CONFIG["standing_water"]["severity"],
                        "color": HAZARD_CONFIG["standing_water"]["color"],
                        "description": HAZARD_CONFIG["standing_water"]["description"]
                    })

        # 3. Detect Potholes (Dark localized depressed regions with edge gradient)
        # Calculate local road brightness baseline
        road_pixels = gray[road_mask == 255]
        if len(road_pixels) > 0:
            median_road_lum = np.median(road_pixels)
            # Cavities are significantly darker than the surrounding road
            dark_thresh = max(15, median_road_lum - 28)
            _, dark_mask = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)
            dark_mask = cv2.bitwise_and(dark_mask, road_mask)
            dark_mask[:horizon_y + 25, :] = 0
            
            kernel_pothole = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7))
            dark_clean = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_pothole)
            dark_clean = cv2.morphologyEx(dark_clean, cv2.MORPH_CLOSE, kernel_pothole)
            
            contours, _ = cv2.findContours(dark_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if 250 < area < (w * h * 0.12):
                    x, y, bw, bh = cv2.boundingRect(c)
                    circularity = 4 * np.pi * (area / (cv2.arcLength(c, True) ** 2 + 1e-5))
                    aspect = bw / float(bh)
                    if 0.6 < aspect < 3.5 and circularity > 0.2:
                        confidence = min(0.94, 0.62 + (area / 1200.0) * 0.08)
                        detections.append({
                            "class": "pothole",
                            "confidence": round(float(confidence), 2),
                            "bbox": [x, y, x + bw, y + bh],
                            "severity": HAZARD_CONFIG["pothole"]["severity"],
                            "color": HAZARD_CONFIG["pothole"]["color"],
                            "description": HAZARD_CONFIG["pothole"]["description"]
                        })

        # 4. Detect Cracks (Thin elongated edge networks)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 110)
        edges = cv2.bitwise_and(edges, road_mask)
        edges[:horizon_y + 30, :] = 0
        
        # Dilate slightly to connect fractured lines
        crack_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(edges, crack_kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            length = cv2.arcLength(c, False)
            area = cv2.contourArea(c)
            if length > 70 and area < 800:
                x, y, bw, bh = cv2.boundingRect(c)
                # Avoid overlapping existing potholes or barriers
                already_covered = False
                for d in detections:
                    dx1, dy1, dx2, dy2 = d["bbox"]
                    if not (x + bw < dx1 or x > dx2 or y + bh < dy1 or y > dy2):
                        already_covered = True
                        break
                if not already_covered:
                    confidence = min(0.88, 0.52 + (length / 200.0) * 0.2)
                    detections.append({
                        "class": "road_crack",
                        "confidence": round(float(confidence), 2),
                        "bbox": [x, y, x + bw, y + bh],
                        "severity": HAZARD_CONFIG["road_crack"]["severity"],
                        "color": HAZARD_CONFIG["road_crack"]["color"],
                        "description": HAZARD_CONFIG["road_crack"]["description"]
                    })

        # 5. Detect Debris (Solid objects with contrast and compact shadow)
        # Look for local contrast deviations not classified as water or pothole
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        _, high_grad = cv2.threshold(grad_norm, 75, 255, cv2.THRESH_BINARY)
        high_grad = cv2.bitwise_and(high_grad, road_mask)
        high_grad[:horizon_y + 40, :] = 0
        
        debris_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        high_grad_clean = cv2.morphologyEx(high_grad, cv2.MORPH_CLOSE, debris_kernel)
        contours, _ = cv2.findContours(high_grad_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            area = cv2.contourArea(c)
            if 150 < area < 1500:
                x, y, bw, bh = cv2.boundingRect(c)
                # Check overlap with existing detections
                overlap = False
                for d in detections:
                    dx1, dy1, dx2, dy2 = d["bbox"]
                    if not (x + bw < dx1 or x > dx2 or y + bh < dy1 or y > dy2):
                        overlap = True
                        break
                if not overlap:
                    confidence = min(0.86, 0.55 + (area / 1000.0) * 0.15)
                    detections.append({
                        "class": "debris",
                        "confidence": round(float(confidence), 2),
                        "bbox": [x, y, x + bw, y + bh],
                        "severity": HAZARD_CONFIG["debris"]["severity"],
                        "color": HAZARD_CONFIG["debris"]["color"],
                        "description": HAZARD_CONFIG["debris"]["description"]
                    })

        # Apply Non-Maximum Suppression (NMS) and confidence filtering
        detections = [d for d in detections if d["confidence"] >= self.min_confidence]
        detections = self._apply_nms(detections)
        
        return {
            "image_shape": [h, w],
            "total_hazards": len(detections),
            "detections": detections
        }

    def _apply_nms(self, detections, iou_thresh=0.35):
        """Standard IoU Non-Maximum Suppression."""
        if not detections:
            return []
            
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        keep = []
        
        while detections:
            top = detections.pop(0)
            keep.append(top)
            rem = []
            x1_a, y1_a, x2_a, y2_a = top["bbox"]
            area_a = (x2_a - x1_a) * (y2_a - y1_a)
            
            for d in detections:
                x1_b, y1_b, x2_b, y2_b = d["bbox"]
                area_b = (x2_b - x1_b) * (y2_b - y1_b)
                
                xx1 = max(x1_a, x1_b)
                yy1 = max(y1_a, y1_b)
                xx2 = min(x2_a, x2_b)
                yy2 = min(y2_a, y2_b)
                
                w_inter = max(0, xx2 - xx1)
                h_inter = max(0, yy2 - yy1)
                inter = w_inter * h_inter
                
                iou = inter / float(area_a + area_b - inter + 1e-6)
                if iou < iou_thresh:
                    rem.append(d)
            detections = rem
            
        return keep

    def render_overlay(self, img_input, result):
        """Draws bounding boxes, hazard labels, and severity indicators on image."""
        if isinstance(img_input, str):
            annotated = cv2.imread(img_input)
        else:
            annotated = img_input.copy()
            
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            color = det["color"]
            label = det["class"].replace("_", " ").title()
            conf = det["confidence"]
            sev = det["severity"]
            
            # Box border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Badge header
            tag = f"{label} ({sev}) {int(conf * 100)}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(annotated, tag, (x1 + 5, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
        return annotated

if __name__ == "__main__":
    detector = CVHazardDetector()
    print("CVHazardDetector initialized.")
