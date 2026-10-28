"""
Flask Web Application & REST API for Road Hazard AI
Featuring:
- Perceptual Image Recognition Memory (never forgets human-verified photos)
- Few-Shot Oversampling Training (allows 2-10 photos to genuinely train the neural network)
- Human-in-the-Loop Active Learning Studio
"""

import os
import io
import time
import base64
import json
import shutil
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np

# Adjust path to import models and scripts
BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(BASE_DIR))

from models.ensemble import RoadHazardEnsemble
from models.cv_detector import HAZARD_CONFIG
from scripts.generate_synthetic_data import build_dataset, generate_scene, CLASS_MAP, CLASSES

app = Flask(__name__, 
            template_folder=str(BASE_DIR / "web" / "templates"),
            static_folder=str(BASE_DIR / "web" / "static"))
CORS(app)

# Initialize ensemble
ensemble = RoadHazardEnsemble()

SAMPLES_DIR = BASE_DIR / "samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

DATASET_DIR = BASE_DIR / "dataset"
VERIFIED_MANIFEST_FILE = DATASET_DIR / "verified_manifest.json"

# Thread-safe training state manager
training_lock = threading.Lock()
training_state = {
    "is_training": False,
    "epoch": 0,
    "total_epochs": 0,
    "progress_pct": 0,
    "loss": "N/A",
    "status_message": "Ready to train",
    "start_time": None,
    "elapsed_seconds": 0,
    "logs": []
}

def get_image_hash(img_bgr):
    """Computes a 256-bit perceptual difference/average hash invariant to compression/rescaling."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        avg = small.mean()
        return "".join("1" if val >= avg else "0" for val in small.flatten())
    except Exception:
        return ""

def hamming_dist(h1, h2):
    if not h1 or not h2 or len(h1) != len(h2):
        return 999
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))

def load_verified_manifest():
    if VERIFIED_MANIFEST_FILE.exists():
        try:
            with open(VERIFIED_MANIFEST_FILE, "r") as f:
                manifest = json.load(f)
                # Ensure each entry has image_hash computed
                updated = False
                for entry in manifest:
                    if "image_hash" not in entry:
                        img_path = DATASET_DIR / "images" / "train" / entry["image_file"]
                        if img_path.exists():
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                entry["image_hash"] = get_image_hash(img)
                                updated = True
                if updated:
                    save_verified_manifest(manifest)
                return manifest
        except Exception:
            return []
    return []

def save_verified_manifest(manifest):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(VERIFIED_MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

@app.route("/")
def index():
    return render_template("index.html", hazard_meta=HAZARD_CONFIG)

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns list of available showcase sample road images."""
    if not SAMPLES_DIR.exists():
        return jsonify([])
    files = [f.name for f in SAMPLES_DIR.glob("*.jpg")]
    return jsonify(sorted(files))

@app.route("/api/samples/<filename>", methods=["GET"])
def serve_sample(filename):
    return send_from_directory(str(SAMPLES_DIR), filename)

@app.route("/api/detect", methods=["POST"])
def detect_hazards():
    """
    Evaluates road hazards.
    Checks Perceptual Memory first: If the user previously verified this exact photo,
    it returns their verified ground-truth boxes with 100% confidence!
    """
    engine_choice = request.args.get("engine", "auto")
    bypass_memory = request.args.get("bypass_memory", "false").lower() == "true"
    
    img = None
    raw_b64_original = None

    if "file" in request.files:
        file = request.files["file"]
        in_memory = file.read()
        nparr = np.frombuffer(in_memory, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        raw_b64_original = base64.b64encode(in_memory).decode("utf-8")
    elif request.is_json:
        data = request.get_json()
        if "sample_filename" in data:
            sample_path = SAMPLES_DIR / data["sample_filename"]
            if sample_path.exists():
                img = cv2.imread(str(sample_path))
                with open(sample_path, "rb") as f:
                    raw_b64_original = base64.b64encode(f.read()).decode("utf-8")
        elif "image_base64" in data:
            raw_b64 = data["image_base64"].split(",")[-1]
            raw_b64_original = raw_b64
            img_bytes = base64.b64decode(raw_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "No valid image provided"}), 400

    h, w = img.shape[:2]

    # Clean unannotated image for the interactive canvas
    _, clean_buf = cv2.imencode(".jpg", img)
    clean_b64 = base64.b64encode(clean_buf).decode("utf-8")

    # 1. PERCEPTUAL MEMORY CHECK: Did the user already verify this image?
    matched_entry = None
    if not bypass_memory:
        upload_hash = get_image_hash(img)
        manifest = load_verified_manifest()

        for entry in manifest:
            saved_hash = entry.get("image_hash", "")
            if saved_hash and hamming_dist(upload_hash, saved_hash) <= 12: # Identical or near-identical
                matched_entry = entry
                break

    if matched_entry:
        # Load user-confirmed ground truth boxes!
        user_detections = []
        hazard_counts = {}
        for b in matched_entry["boxes"]:
            c = b["class"]
            hazard_counts[c] = hazard_counts.get(c, 0) + 1
            meta = HAZARD_CONFIG.get(c, {
                "color": [0, 255, 0],
                "severity": "WARNING",
                "description": "User verified road hazard."
            })
            user_detections.append({
                "class": c,
                "confidence": 1.0, # Human confirmed
                "bbox": b["bbox"],
                "severity": meta["severity"],
                "color": meta["color"],
                "description": meta["description"],
                "source": "Human-Verified Ground Truth"
            })

        # Calculate Risk
        weights = {"CRITICAL": 35, "HIGH": 25, "MEDIUM": 15, "WARNING": 10}
        risk_score = min(100, sum(weights.get(d["severity"], 10) for d in user_detections))
        condition = "DANGEROUS" if risk_score > 60 else "CAUTION ADVISED" if risk_score > 30 else "FAIR" if risk_score > 0 else "CLEAR & SAFE"
        cond_color = "#ef4444" if risk_score > 60 else "#f59e0b" if risk_score > 30 else "#3b82f6" if risk_score > 0 else "#10b981"

        return jsonify({
            "status": "success",
            "is_verified_by_user": True,
            "verified_sample_id": matched_entry["id"],
            "result": {
                "total_hazards": len(user_detections),
                "risk_score": risk_score,
                "condition": f"VERIFIED: {condition}",
                "condition_color": "#10b981", # Emerald
                "engine": "Human-Verified Ground Truth Memory",
                "detections": user_detections
            },
            "hazard_counts": hazard_counts,
            "recommendations": [{"priority": "VERIFIED", "action": "This photo contains your confirmed, verified hazard annotations."}],
            "raw_detections": user_detections,
            "image_dimensions": {"width": w, "height": h},
            "clean_image_b64": f"data:image/jpeg;base64,{clean_b64}",
            "original_image_b64": f"data:image/jpeg;base64,{clean_b64}",
            "annotated_image": f"data:image/jpeg;base64,{clean_b64}"
        })

    # 2. STANDARD AI INFERENCE (For new unseen photos)
    result = ensemble.analyze(img, engine=engine_choice)
    
    hazard_counts = {}
    for d in result["detections"]:
        c = d["class"]
        hazard_counts[c] = hazard_counts.get(c, 0) + 1

    recommendations = []
    if hazard_counts.get("pothole", 0) > 0:
        recommendations.append({
            "priority": "HIGH",
            "action": f"Dispatch highway repair crew for {hazard_counts['pothole']} pothole(s). Cold/Hot mix asphalt patching recommended immediately."
        })
    if hazard_counts.get("road_crack", 0) > 0:
        recommendations.append({
            "priority": "MEDIUM",
            "action": f"Seal {hazard_counts['road_crack']} crack segment(s) with bitumen crack sealant to prevent base saturation."
        })
    if hazard_counts.get("debris", 0) > 0:
        recommendations.append({
            "priority": "CRITICAL",
            "action": f"Issue lane obstacle warning. Obstacle clearance team needed to remove road debris."
        })
    if hazard_counts.get("standing_water", 0) > 0:
        recommendations.append({
            "priority": "HIGH",
            "action": f"Aquaplaning risk detected ({hazard_counts['standing_water']} puddle areas). Check roadside drainage gratings."
        })
    if hazard_counts.get("construction_barrier", 0) > 0:
        recommendations.append({
            "priority": "INFO",
            "action": "Active construction barrier detected. Enforce work-zone speed reduction (30 km/h)."
        })

    if not recommendations:
        recommendations.append({
            "priority": "OPTIMAL",
            "action": "Road surface is in good driving condition. No immediate interventions required."
        })

    return jsonify({
        "status": "success",
        "is_verified_by_user": False,
        "result": result,
        "hazard_counts": hazard_counts,
        "recommendations": recommendations,
        "raw_detections": result["detections"],
        "image_dimensions": {"width": w, "height": h},
        "clean_image_b64": f"data:image/jpeg;base64,{clean_b64}",
        "original_image_b64": f"data:image/jpeg;base64,{clean_b64}",
        "annotated_image": f"data:image/jpeg;base64,{clean_b64}"
    })

@app.route("/api/dataset/submit-verified", methods=["POST"])
def submit_verified_sample():
    """Saves human-verified image + bounding boxes into the verified dataset & manifest."""
    data = request.get_json() or {}
    raw_b64 = data.get("image_base64", "")
    boxes = data.get("boxes", [])

    if not raw_b64:
        return jsonify({"error": "Missing image"}), 400

    raw_bytes = base64.b64decode(raw_b64.split(",")[-1])
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image data"}), 400

    h, w = img.shape[:2]
    img_hash = get_image_hash(img)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    sample_id = f"verified_{timestamp_str}"

    train_img_dir = DATASET_DIR / "images" / "train"
    train_lbl_dir = DATASET_DIR / "labels" / "train"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_filename = f"{sample_id}.jpg"
    lbl_filename = f"{sample_id}.txt"

    cv2.imwrite(str(train_img_dir / img_filename), img)

    valid_boxes = []
    with open(train_lbl_dir / lbl_filename, "w") as f:
        for b in boxes:
            cname = b.get("class", "pothole").lower()
            cid = CLASS_MAP.get(cname, 0)
            coords = b.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(v) for v in coords]
            
            bw = (x2 - x1) / float(w)
            bh = (y2 - y1) / float(h)
            bx = (x1 + x2) / (2.0 * w)
            by = (y1 + y2) / (2.0 * h)

            if bw > 0.005 and bh > 0.005:
                f.write(f"{cid} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
                valid_boxes.append({
                    "class": cname,
                    "bbox": [x1, y1, x2, y2],
                    "norm": [bx, by, bw, bh]
                })

    manifest = load_verified_manifest()
    
    # If same image was previously verified, update it
    existing_idx = None
    for idx, e in enumerate(manifest):
        if e.get("image_hash") and hamming_dist(img_hash, e["image_hash"]) <= 12:
            existing_idx = idx
            break

    entry = {
        "id": sample_id,
        "image_file": img_filename,
        "image_hash": img_hash,
        "timestamp": datetime.now().isoformat(),
        "hazards_count": len(valid_boxes),
        "classes": list(set(b["class"] for b in valid_boxes)),
        "boxes": valid_boxes
    }

    if existing_idx is not None:
        manifest[existing_idx] = entry
    else:
        manifest.insert(0, entry)

    save_verified_manifest(manifest)

    return jsonify({
        "status": "success",
        "message": f"Saved! Human-verified sample logged with {len(valid_boxes)} confirmed hazard(s).",
        "sample_id": sample_id,
        "total_verified": len(manifest)
    })

@app.route("/api/stats", methods=["GET"])
def dataset_stats():
    """Returns local dataset and active model statistics."""
    dataset_dir = BASE_DIR / "dataset"
    train_images = len(list((dataset_dir / "images" / "train").glob("*.jpg"))) if (dataset_dir / "images" / "train").exists() else 0
    val_images = len(list((dataset_dir / "images" / "val").glob("*.jpg"))) if (dataset_dir / "images" / "val").exists() else 0
    
    custom_weights = BASE_DIR / "models" / "road_hazard_best.pt"
    rdd_weights = BASE_DIR / "models" / "road_damage_yolov8.pt"

    active_type = ensemble.yolo_detector.model_type if ensemble.yolo_detector else "None"
    if active_type == "fine_tuned":
        display_name = "My Custom Fine-Tuned Model (road_hazard_best.pt)"
    elif active_type == "rdd_real_world":
        display_name = "Real-World RDD Model (Trained on 26,000+ Real Roads)"
    else:
        display_name = "YOLOv8 Base (yolov8n.pt)"

    manifest = load_verified_manifest()

    return jsonify({
        "train_samples": train_images,
        "val_samples": val_images,
        "verified_human_samples": len(manifest),
        "classes": list(HAZARD_CONFIG.keys()),
        "custom_weights_trained": custom_weights.exists(),
        "rdd_weights_available": rdd_weights.exists(),
        "active_model_name": display_name,
        "active_model_type": active_type,
        "is_training": training_state["is_training"]
    })

@app.route("/api/model/switch", methods=["POST"])
def switch_model():
    """Switches the active model in memory between 'rdd', 'custom', or 'base'."""
    data = request.get_json() or {}
    target = data.get("model_type", "rdd")
    
    custom_weights = BASE_DIR / "models" / "road_hazard_best.pt"
    rdd_weights = BASE_DIR / "models" / "road_damage_yolov8.pt"
    
    if target == "custom" and custom_weights.exists():
        ensemble.yolo_detector.weights_path = str(custom_weights)
        ensemble.yolo_detector._load_model()
        msg = "Switched to Custom Fine-Tuned Model"
    elif target == "rdd" and rdd_weights.exists():
        ensemble.yolo_detector.weights_path = str(rdd_weights)
        ensemble.yolo_detector._load_model()
        msg = "Switched to Real-World RDD Model"
    else:
        ensemble.yolo_detector.weights_path = None
        ensemble.yolo_detector._load_model()
        msg = "Switched to Base YOLO Model"

    return jsonify({"status": "success", "message": msg, "active_type": ensemble.yolo_detector.model_type})

def _training_worker(epochs=10, batch_size=4, imgsz=512):
    """
    Background worker executing model training with Few-Shot Oversampling.
    The user's verified images are oversampled 25x so the neural network heavily
    updates on human corrections even when only 2 or 5 photos are provided!
    """
    global training_state
    
    with training_lock:
        training_state["is_training"] = True
        training_state["epoch"] = 0
        training_state["total_epochs"] = epochs
        training_state["progress_pct"] = 0
        training_state["status_message"] = "Initializing training pipeline..."
        training_state["start_time"] = time.time()
        training_state["logs"] = ["Training initialized..."]

    try:
        from ultralytics import YOLO
        data_yaml = BASE_DIR / "dataset" / "data.yaml"
        manifest = load_verified_manifest()
        verified_count = len(manifest)

        # Build dynamic training list with 25x Few-Shot Oversampling
        train_img_dir = DATASET_DIR / "images" / "train"
        train_list_file = DATASET_DIR / "train_active_list.txt"
        
        all_train_files = list(train_img_dir.glob("*.jpg"))
        verified_files = [train_img_dir / e["image_file"] for e in manifest if (train_img_dir / e["image_file"]).exists()]
        baseline_files = [f for f in all_train_files if not f.name.startswith("verified_")]

        oversampled_list = []
        # Add baseline samples (subsample to max 40 so they don't drown user's photos)
        oversampled_list.extend([str(f).replace(chr(92), '/') for f in baseline_files[:40]])

        # Oversample user-verified photos 25 times each!
        OVERSAMPLE_FACTOR = 25
        for vf in verified_files:
            v_str = str(vf).replace(chr(92), '/')
            for _ in range(OVERSAMPLE_FACTOR):
                oversampled_list.append(v_str)

        with open(train_list_file, "w") as f:
            for item in oversampled_list:
                f.write(f"{item}\n")

        # Create active data.yaml pointing to the balanced oversampled training list
        active_yaml = DATASET_DIR / "data_active.yaml"
        active_yaml_content = f"""path: {str(DATASET_DIR).replace(chr(92), '/')}
train: train_active_list.txt
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
        with open(active_yaml, "w") as f:
            f.write(active_yaml_content)

        with training_lock:
            if verified_count > 0:
                training_state["logs"].append(f"Applying Few-Shot Oversampling: {verified_count} verified photo(s) boosted 25x in training set!")
            else:
                training_state["logs"].append("Training on baseline dataset...")

        model = YOLO("yolov8n.pt")

        def on_train_epoch_end(trainer):
            with training_lock:
                cur_ep = trainer.epoch + 1
                tot_ep = trainer.epochs
                training_state["epoch"] = cur_ep
                training_state["progress_pct"] = round((cur_ep / float(tot_ep)) * 100, 1)
                training_state["elapsed_seconds"] = int(time.time() - training_state["start_time"])
                
                loss_val = "N/A"
                if hasattr(trainer, "loss") and trainer.loss is not None:
                    loss_val = f"{float(trainer.loss.item() if hasattr(trainer.loss, 'item') else trainer.loss):.4f}"
                training_state["loss"] = loss_val
                
                log_entry = f"Epoch {cur_ep}/{tot_ep} completed • Loss: {loss_val}"
                training_state["status_message"] = log_entry
                training_state["logs"].append(log_entry)

        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        runs_dir = BASE_DIR / "runs" / "web_train"
        with training_lock:
            training_state["status_message"] = "Fine-tuning neural network on verified hazard data..."
            training_state["logs"].append("Neural weights optimizing...")

        model.train(
            data=str(active_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=str(runs_dir),
            name="web_model",
            exist_ok=True,
            verbose=False
        )

        best_pt = runs_dir / "web_model" / "weights" / "best.pt"
        target_pt = BASE_DIR / "models" / "road_hazard_best.pt"
        if best_pt.exists():
            shutil.copy(best_pt, target_pt)
            ensemble.yolo_detector.weights_path = str(target_pt)
            ensemble.yolo_detector._load_model()

        with training_lock:
            training_state["is_training"] = False
            training_state["progress_pct"] = 100
            training_state["status_message"] = "Training successfully completed! New model active."
            training_state["logs"].append("New custom model fine-tuned and active in vision pipeline!")

    except Exception as e:
        with training_lock:
            training_state["is_training"] = False
            training_state["status_message"] = f"Training error: {str(e)}"
            training_state["logs"].append(f"ERROR: {str(e)}")

@app.route("/api/train/start", methods=["POST"])
def start_training():
    """Initiates an asynchronous model training session."""
    if training_state["is_training"]:
        return jsonify({"status": "busy", "message": "A training session is already in progress!"}), 400

    data = request.get_json() or {}
    epochs = int(data.get("epochs", 5))
    batch_size = int(data.get("batch_size", 4))
    imgsz = int(data.get("imgsz", 512))

    t = threading.Thread(target=_training_worker, kwargs={"epochs": epochs, "batch_size": batch_size, "imgsz": imgsz})
    t.daemon = True
    t.start()

    return jsonify({
        "status": "started",
        "message": f"Training initiated on verified dataset ({epochs} epochs, batch {batch_size})."
    })

@app.route("/api/train/status", methods=["GET"])
def get_training_status():
    """Returns real-time training progress for the frontend HUD."""
    with training_lock:
        copy_state = dict(training_state)
        copy_state["logs"] = copy_state["logs"][-8:]
    return jsonify(copy_state)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
