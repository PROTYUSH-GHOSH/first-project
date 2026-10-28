"""
Train YOLOv8 on Road Hazard Dataset
Supports training on custom synthetic or annotated real-world road datasets.
"""

import os
import sys
import argparse
from pathlib import Path

def train_hazard_model(data_yaml=None, epochs=20, imgsz=640, batch_size=8, device='cpu', base_model='yolov8n.pt'):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics package is not yet available. Please run: pip install ultralytics")
        return False

    proj_root = Path(__file__).resolve().parent.parent
    if data_yaml is None:
        data_yaml = proj_root / "dataset" / "data.yaml"

    if not os.path.exists(data_yaml):
        print(f"Error: dataset YAML config not found at {data_yaml}")
        print("Please run scripts/generate_synthetic_data.py first to generate the dataset!")
        return False

    print("=" * 60)
    print("Road Hazard AI - Model Training Engine")
    print("=" * 60)
    print(f"Dataset Config : {data_yaml}")
    print(f"Base Model     : {base_model}")
    print(f"Epochs         : {epochs}")
    print(f"Batch Size     : {batch_size}")
    print(f"Image Size     : {imgsz}")
    print(f"Device         : {device}")
    print("=" * 60)

    model = YOLO(base_model)

    runs_dir = proj_root / "runs" / "train"
    os.makedirs(runs_dir, exist_ok=True)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=str(runs_dir),
        name="hazard_detector",
        exist_ok=True,
        verbose=True
    )

    best_weights = runs_dir / "hazard_detector" / "weights" / "best.pt"
    target_weights = proj_root / "models" / "road_hazard_best.pt"
    
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, target_weights)
        print(f"\nModel training finished successfully!")
        print(f"Best model weights saved to: {target_weights}")
    else:
        print(f"\nTraining completed. Weights available under: {runs_dir}")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Road Hazard YOLO Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or 0 for GPU)")
    args = parser.parse_args()

    train_hazard_model(epochs=args.epochs, batch_size=args.batch, imgsz=args.imgsz, device=args.device)
