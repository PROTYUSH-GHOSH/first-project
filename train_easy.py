"""
RoadGuard AI - 1-Click Easy Trainer
Fine-tunes a custom neural network on your road hazard dataset with smart defaults.
Usage:
    python train_easy.py
    python train_easy.py --epochs 15 --batch 4
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="1-Click Easy Road Hazard Model Trainer")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default: 4 for smooth CPU/GPU training)")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size (default: 512)")
    parser.add_argument("--base", type=str, default="yolov8n.pt", help="Base model (default: yolov8n.pt)")
    args = parser.parse_args()

    proj_dir = Path(__file__).resolve().parent
    data_yaml = proj_dir / "dataset" / "data.yaml"

    if not data_yaml.exists():
        print("[!] Dataset data.yaml not found. Generating synthetic dataset first...")
        from scripts.generate_synthetic_data import build_dataset
        build_dataset(str(proj_dir), train_count=80, val_count=20, sample_count=6)

    print("=" * 65)
    print("🚦 RoadGuard AI - 1-Click Model Fine-Tuning")
    print("=" * 65)
    print(f"📁 Dataset Config   : {data_yaml}")
    print(f"🧠 Base Checkpoint  : {args.base}")
    print(f"🔄 Epochs           : {args.epochs}")
    print(f"📦 Batch Size       : {args.batch}")
    print(f"🖼️ Image Resolution : {args.imgsz}x{args.imgsz}")
    print("=" * 65)
    print("\nStarting Transfer Learning (Press Ctrl+C at any time to stop)...\n")

    model = YOLO(args.base)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(proj_dir / "runs" / "train"),
        name="hazard_model",
        exist_ok=True,
        verbose=True
    )

    # Automatically copy the trained best.pt into models/road_hazard_best.pt
    best_weights = proj_dir / "runs" / "train" / "hazard_model" / "weights" / "best.pt"
    target_weights = proj_dir / "models" / "road_hazard_best.pt"

    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, target_weights)
        print("\n" + "=" * 65)
        print("🎉 SUCCESS! Your new custom AI model has been trained and activated!")
        print(f"💾 Saved to: {target_weights}")
        print("The web dashboard will automatically use your new custom model on its next prediction.")
        print("=" * 65)
    else:
        print("\nTraining completed. Weights available in runs/train/hazard_model/weights/")

if __name__ == "__main__":
    main()
