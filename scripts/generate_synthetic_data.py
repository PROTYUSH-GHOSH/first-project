"""
Synthetic Road Hazard Dataset Generator
Generates realistic perspective road scenes with embedded hazards:
- Potholes (depth, inner shadow, crumbling edges)
- Road Cracks (branching fractures, alligator cracking)
- Road Debris (fallen cargo, tire treads, rocks)
- Standing Water (puddle reflections, specular highlights)
- Construction Barriers (traffic cones, striped barricades)

Exports directly into YOLOv8 annotation format (images + labels .txt).
"""

import os
import random
import math
import numpy as np
import cv2

CLASSES = [
    "pothole",
    "road_crack",
    "debris",
    "standing_water",
    "construction_barrier"
]

CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}

def create_base_road(width=800, height=600):
    """Generates an asphalt road scene with vanishing point perspective and lane markings."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 1. Sky & Horizon (top 30%)
    horizon_y = int(height * 0.35)
    for y in range(horizon_y):
        sky_ratio = y / horizon_y
        b = int(220 - 40 * sky_ratio)
        g = int(180 - 30 * sky_ratio)
        r = int(140 - 20 * sky_ratio)
        img[y, :] = (b, g, r)
        
    # Horizon line / background greenery / trees
    cv2.rectangle(img, (0, horizon_y - 20), (width, horizon_y), (40, 70, 40), -1)
    
    # 2. Road surface with asphalt gradient and texture
    road_top_left = (int(width * 0.38), horizon_y)
    road_top_right = (int(width * 0.62), horizon_y)
    road_bottom_left = (int(width * 0.05), height)
    road_bottom_right = (int(width * 0.95), height)
    
    # Shoulders / grass outside road
    grass_color = (45, 85, 45)
    img[horizon_y:height, :] = grass_color
    
    # Road polygon
    road_pts = np.array([road_top_left, road_top_right, road_bottom_right, road_bottom_left], dtype=np.int32)
    road_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(road_mask, [road_pts], 255)
    
    # Road base asphalt with vertical light falloff
    asphalt_layer = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(horizon_y, height):
        progress = (y - horizon_y) / (height - horizon_y)
        # Closer road is darker & more detailed
        base_gray = int(75 - 20 * progress + random.randint(-4, 4))
        base_gray = max(35, min(120, base_gray))
        asphalt_layer[y, :] = (base_gray - 3, base_gray, base_gray + 2)
        
    # Asphalt noise / grain
    grain = np.random.normal(0, 10, (height, width, 3)).astype(np.int16)
    asphalt_layer = np.clip(asphalt_layer.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    
    # Combine road into canvas
    img[road_mask == 255] = asphalt_layer[road_mask == 255]
    
    # 3. Lane Markings
    # Center dashed yellow line
    steps = 16
    for i in range(steps):
        if i % 2 == 0:
            continue
        p1 = i / steps
        p2 = (i + 0.75) / steps
        y1 = int(horizon_y + p1 * (height - horizon_y))
        y2 = int(horizon_y + p2 * (height - horizon_y))
        cx1 = int(road_top_left[0] + (road_top_right[0] - road_top_left[0]) * 0.5)
        cx2 = int(road_bottom_left[0] + (road_bottom_right[0] - road_bottom_left[0]) * 0.5)
        
        x1 = int(cx1 + (cx2 - cx1) * p1)
        x2 = int(cx1 + (cx2 - cx1) * p2)
        thickness = max(2, int(p2 * 7))
        cv2.line(img, (x1, y1), (x2, y2), (25, 210, 240), thickness)
        
    # White shoulder lines
    for side in [0.08, 0.92]:
        x_top = int(road_top_left[0] + (road_top_right[0] - road_top_left[0]) * side)
        x_bot = int(road_bottom_left[0] + (road_bottom_right[0] - road_bottom_left[0]) * side)
        for i in range(steps):
            p1 = i / steps
            p2 = (i + 1.0) / steps
            y1 = int(horizon_y + p1 * (height - horizon_y))
            y2 = int(horizon_y + p2 * (height - horizon_y))
            x1 = int(x_top + (x_bot - x_top) * p1)
            x2 = int(x_top + (x_bot - x_top) * p2)
            thickness = max(2, int(p2 * 6))
            cv2.line(img, (x1, y1), (x2, y2), (220, 220, 225), thickness)

    return img, horizon_y

def draw_pothole(img, center, size):
    """Draws a photorealistic pothole with deep hollow, crumbling edge, and asphalt rubble."""
    cx, cy = center
    rw, rh = size
    
    # Irregular polygon for pothole rim
    angles = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    pts = []
    for a in angles:
        r = 1.0 + random.uniform(-0.25, 0.25)
        px = int(cx + rw * 0.5 * r * math.cos(a))
        py = int(cy + rh * 0.35 * r * math.sin(a)) # compressed perspective
        pts.append([px, py])
    pts = np.array(pts, dtype=np.int32)
    
    # Crater shadow & dark cavity
    cavity_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(cavity_mask, [pts], 255)
    
    # Dark cavity fill
    dark_fill = np.zeros_like(img)
    dark_fill[:] = (18, 18, 20)
    # Add subtle interior stone gradient
    cv2.ellipse(dark_fill, (cx, cy + int(rh * 0.08)), (int(rw * 0.4), int(rh * 0.25)), 0, 0, 360, (35, 35, 38), -1)
    
    img[cavity_mask == 255] = cv2.addWeighted(img[cavity_mask == 255], 0.15, dark_fill[cavity_mask == 255], 0.85, 0)
    
    # Jagged crumbled rim
    cv2.polylines(img, [pts], isClosed=True, color=(15, 15, 15), thickness=3)
    cv2.polylines(img, [pts], isClosed=True, color=(120, 120, 125), thickness=1)
    
    # Small rubble pebbles around pothole
    for _ in range(random.randint(6, 15)):
        ox = cx + random.randint(-int(rw * 0.65), int(rw * 0.65))
        oy = cy + random.randint(-int(rh * 0.45), int(rh * 0.45))
        peb_r = random.randint(2, 4)
        cv2.circle(img, (ox, oy), peb_r, (40, 40, 42), -1)
        cv2.circle(img, (ox - 1, oy - 1), max(1, peb_r - 1), (110, 110, 115), -1)
        
    # Return bounding box
    x_min = max(0, min(p[0] for p in pts) - 3)
    x_max = min(img.shape[1], max(p[0] for p in pts) + 3)
    y_min = max(0, min(p[1] for p in pts) - 3)
    y_max = min(img.shape[0], max(p[1] for p in pts) + 3)
    return [x_min, y_min, x_max, y_max]

def draw_road_crack(img, start, length):
    """Draws branching longitudinal/transverse alligator road crack."""
    x, y = start
    cur_x, cur_y = x, y
    min_x, max_x = cur_x, cur_x
    min_y, max_y = cur_y, cur_y
    
    segments = random.randint(8, 18)
    step_len = length / segments
    
    for _ in range(segments):
        dx = random.uniform(-step_len * 1.2, step_len * 1.2)
        dy = random.uniform(step_len * 0.5, step_len * 1.4)
        next_x = int(np.clip(cur_x + dx, 10, img.shape[1] - 10))
        next_y = int(np.clip(cur_y + dy, 10, img.shape[0] - 10))
        
        # Draw dark crack line with subtle gray outline
        cv2.line(img, (cur_x, cur_y), (next_x, next_y), (20, 20, 20), 2)
        cv2.line(img, (cur_x + 1, cur_y), (next_x + 1, next_y), (90, 90, 95), 1)
        
        # Branching sub-crack
        if random.random() < 0.45:
            b_dx = random.uniform(-25, 25)
            b_dy = random.uniform(5, 25)
            bx = int(np.clip(cur_x + b_dx, 10, img.shape[1] - 10))
            by = int(np.clip(cur_y + b_dy, 10, img.shape[0] - 10))
            cv2.line(img, (cur_x, cur_y), (bx, by), (25, 25, 25), 1)
            min_x = min(min_x, bx)
            max_x = max(max_x, bx)
            min_y = min(min_y, by)
            max_y = max(max_y, by)
            
        min_x = min(min_x, cur_x, next_x)
        max_x = max(max_x, cur_x, next_x)
        min_y = min(min_y, cur_y, next_y)
        max_y = max(max_y, cur_y, next_y)
        
        cur_x, cur_y = next_x, next_y
        
    return [max(0, min_x - 5), max(0, min_y - 5), min(img.shape[1], max_x + 5), min(img.shape[0], max_y + 5)]

def draw_debris(img, center, size):
    """Draws road debris (tire scrap, wood block, rock, lost cargo box)."""
    cx, cy = center
    w, h = size
    debris_type = random.choice(["tire_scrap", "wood_cargo", "rock_boulder"])
    
    # Shadow under object
    cv2.ellipse(img, (cx + 4, cy + int(h * 0.45)), (int(w * 0.55), int(h * 0.2)), 0, 0, 360, (20, 20, 20), -1)
    
    if debris_type == "tire_scrap":
        # Curving black tire tread
        pts = np.array([
            [cx - int(w * 0.5), cy + int(h * 0.2)],
            [cx - int(w * 0.3), cy - int(h * 0.4)],
            [cx + int(w * 0.3), cy - int(h * 0.3)],
            [cx + int(w * 0.5), cy + int(h * 0.3)],
            [cx + int(w * 0.2), cy + int(h * 0.4)],
            [cx - int(w * 0.2), cy + int(h * 0.1)]
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], (22, 22, 22))
        cv2.polylines(img, [pts], True, (45, 45, 48), 2)
        # Tire tread ridges
        for i in range(-2, 3):
            cv2.line(img, (cx + i * 8, cy - 8), (cx + i * 8 + 5, cy + 10), (55, 55, 58), 1)
            
    elif debris_type == "wood_cargo":
        # Brown rectangular wooden box / pallet fragment
        x1 = cx - int(w * 0.45)
        y1 = cy - int(h * 0.45)
        x2 = cx + int(w * 0.45)
        y2 = cy + int(h * 0.4)
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 75, 115), -1) # BGR brownish
        cv2.rectangle(img, (x1, y1), (x2, y2), (25, 45, 75), 2)
        # Cross braces
        cv2.line(img, (x1, y1), (x2, y2), (30, 55, 90), 2)
        cv2.line(img, (x1, y2), (x2, y1), (30, 55, 90), 2)
        
    else: # rock_boulder
        pts = []
        for a in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            r = 0.8 + random.uniform(-0.25, 0.25)
            px = int(cx + w * 0.45 * r * math.cos(a))
            py = int(cy + h * 0.4 * r * math.sin(a))
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(img, [pts], (80, 85, 90))
        cv2.polylines(img, [pts], True, (50, 52, 55), 2)
        # Highlight facet
        cv2.line(img, (cx - 5, cy - 5), (cx + 5, cy + 5), (120, 125, 130), 2)

    return [max(0, cx - int(w * 0.55)), max(0, cy - int(h * 0.55)), min(img.shape[1], cx + int(w * 0.55)), min(img.shape[0], cy + int(h * 0.55))]

def draw_standing_water(img, center, size):
    """Draws a standing water puddle with sky reflection and wet asphalt ring."""
    cx, cy = center
    rw, rh = size
    
    # Outer wet darkening ring
    cv2.ellipse(img, (cx, cy), (int(rw * 0.55), int(rh * 0.35)), 0, 0, 360, (25, 30, 32), -1)
    
    # Specular water reflection
    overlay = img.copy()
    cv2.ellipse(overlay, (cx, cy), (int(rw * 0.45), int(rh * 0.25)), 0, 0, 360, (140, 160, 175), -1)
    # Bright sky glint inside puddle
    cv2.ellipse(overlay, (cx + int(rw * 0.1), cy - int(rh * 0.05)), (int(rw * 0.25), int(rh * 0.1)), -15, 0, 360, (210, 225, 235), -1)
    
    img[:] = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    return [max(0, cx - int(rw * 0.55)), max(0, cy - int(rh * 0.35)), min(img.shape[1], cx + int(rw * 0.55)), min(img.shape[0], cy + int(rh * 0.35))]

def draw_construction_barrier(img, center, size):
    """Draws a traffic cone or striped construction barricade."""
    cx, cy = center
    w, h = size
    is_cone = random.choice([True, False])
    
    # Cast shadow
    cv2.ellipse(img, (cx + 6, cy + int(h * 0.45)), (int(w * 0.5), int(h * 0.15)), 0, 0, 360, (20, 20, 20), -1)
    
    if is_cone:
        # Traffic cone: Base + triangle + white reflective bands
        base_y = cy + int(h * 0.45)
        top_y = cy - int(h * 0.5)
        cv2.rectangle(img, (cx - int(w * 0.45), base_y - 6), (cx + int(w * 0.45), base_y + 4), (10, 80, 220), -1) # Orange base
        
        # Triangle
        cone_pts = np.array([
            [cx, top_y],
            [cx + int(w * 0.35), base_y],
            [cx - int(w * 0.35), base_y]
        ], dtype=np.int32)
        cv2.fillPoly(img, [cone_pts], (15, 100, 240))
        
        # White reflective stripes
        for band_pos in [0.4, 0.7]:
            by = int(top_y + (base_y - top_y) * band_pos)
            bw = int(w * 0.7 * band_pos * 0.5)
            cv2.line(img, (cx - bw, by), (cx + bw, by), (240, 240, 245), max(2, int(h * 0.08)))
    else:
        # Striped barricade / drum
        x1 = cx - int(w * 0.4)
        x2 = cx + int(w * 0.4)
        y1 = cy - int(h * 0.5)
        y2 = cy + int(h * 0.45)
        # Alternating orange/white diagonal stripes
        cv2.rectangle(img, (x1, y1), (x2, y2), (15, 100, 240), -1)
        stripe_height = (y2 - y1) // 4
        for s in range(4):
            if s % 2 == 1:
                cv2.rectangle(img, (x1, y1 + s * stripe_height), (x2, y1 + (s + 1) * stripe_height), (240, 240, 245), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)
        
    return [max(0, cx - int(w * 0.5)), max(0, cy - int(h * 0.55)), min(img.shape[1], cx + int(w * 0.5)), min(img.shape[0], cy + int(h * 0.55))]

def generate_scene(width=800, height=600, min_hazards=1, max_hazards=4):
    """Generates a complete scene with road and hazards with YOLO ground truth."""
    img, horizon_y = create_base_road(width, height)
    num_hazards = random.randint(min_hazards, max_hazards)
    
    boxes = []
    
    for _ in range(num_hazards):
        hazard_type = random.choice(CLASSES)
        class_id = CLASS_MAP[hazard_type]
        
        # Place hazard within the visible road boundaries
        y_pos = random.randint(int(horizon_y + 40), height - 40)
        progress = (y_pos - horizon_y) / (height - horizon_y)
        
        # Scale objects realistically with perspective (smaller near horizon, larger close)
        scale = 0.35 + progress * 0.95
        
        # Road width at this y
        left_bound = int(width * 0.38 + (width * 0.05 - width * 0.38) * progress + 20)
        right_bound = int(width * 0.62 + (width * 0.95 - width * 0.62) * progress - 20)
        
        if right_bound <= left_bound + 30:
            continue
            
        x_pos = random.randint(left_bound + 15, right_bound - 15)
        
        if hazard_type == "pothole":
            w = int(random.randint(60, 110) * scale)
            h = int(random.randint(40, 75) * scale)
            box = draw_pothole(img, (x_pos, y_pos), (w, h))
        elif hazard_type == "road_crack":
            length = random.randint(60, 160) * scale
            box = draw_crack = draw_road_crack(img, (x_pos, y_pos), length)
            box = draw_crack
        elif hazard_type == "debris":
            w = int(random.randint(35, 75) * scale)
            h = int(random.randint(25, 55) * scale)
            box = draw_debris(img, (x_pos, y_pos), (w, h))
        elif hazard_type == "standing_water":
            w = int(random.randint(80, 160) * scale)
            h = int(random.randint(45, 85) * scale)
            box = draw_standing_water(img, (x_pos, y_pos), (w, h))
        elif hazard_type == "construction_barrier":
            w = int(random.randint(35, 60) * scale)
            h = int(random.randint(60, 100) * scale)
            box = draw_construction_barrier(img, (x_pos, y_pos), (w, h))
            
        x1, y1, x2, y2 = box
        # Convert to YOLO format: class_id, x_center, y_center, width, height (normalized 0-1)
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        bx = (x1 + x2) / (2.0 * width)
        by = (y1 + y2) / (2.0 * height)
        
        if bw > 0.01 and bh > 0.01:
            boxes.append((class_id, bx, by, bw, bh, [x1, y1, x2, y2]))
            
    return img, boxes

def build_dataset(base_dir, train_count=120, val_count=30, sample_count=10):
    """Creates the dataset folder structure and writes images and annotations."""
    train_img_dir = os.path.join(base_dir, "dataset", "images", "train")
    val_img_dir = os.path.join(base_dir, "dataset", "images", "val")
    train_lbl_dir = os.path.join(base_dir, "dataset", "labels", "train")
    val_lbl_dir = os.path.join(base_dir, "dataset", "labels", "val")
    sample_dir = os.path.join(base_dir, "samples")
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir, sample_dir]:
        os.makedirs(d, exist_ok=True)
        
    print(f"Generating {train_count} training samples...")
    for i in range(train_count):
        img, boxes = generate_scene()
        img_name = f"road_train_{i:04d}.jpg"
        lbl_name = f"road_train_{i:04d}.txt"
        cv2.imwrite(os.path.join(train_img_dir, img_name), img)
        with open(os.path.join(train_lbl_dir, lbl_name), "w") as f:
            for b in boxes:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                
    print(f"Generating {val_count} validation samples...")
    for i in range(val_count):
        img, boxes = generate_scene()
        img_name = f"road_val_{i:04d}.jpg"
        lbl_name = f"road_val_{i:04d}.txt"
        cv2.imwrite(os.path.join(val_img_dir, img_name), img)
        with open(os.path.join(val_lbl_dir, lbl_name), "w") as f:
            for b in boxes:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                
    print(f"Generating {sample_count} standalone showcase test samples...")
    for i in range(sample_count):
        img, boxes = generate_scene()
        img_name = f"sample_road_{i+1:02d}.jpg"
        cv2.imwrite(os.path.join(sample_dir, img_name), img)

    # Generate data.yaml for YOLO training
    yaml_path = os.path.join(base_dir, "dataset", "data.yaml")
    data_yaml_content = f"""path: {os.path.abspath(os.path.join(base_dir, 'dataset')).replace(chr(92), '/')}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(yaml_path, "w") as f:
        f.write(data_yaml_content)
        
    print(f"Dataset generated successfully at: {os.path.join(base_dir, 'dataset')}")
    print(f"YAML config created at: {yaml_path}")

if __name__ == "__main__":
    import sys
    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Target Project Directory: {proj_dir}")
    build_dataset(proj_dir, train_count=100, val_count=25, sample_count=8)
