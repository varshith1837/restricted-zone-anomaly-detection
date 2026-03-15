"""
Sentinel Vision — YOLOv8 Fine-Tuning Script
=============================================
Fine-tunes YOLOv8 on surveillance / indoor person detection data.

This script demonstrates:
- Transfer learning from pre-trained YOLOv8 weights
- Custom dataset configuration
- Training hyperparameter management
- mAP evaluation and loss curve logging

Usage:
    python training/fine_tune_yolo.py
    python training/fine_tune_yolo.py --dataset path/to/data.yaml --epochs 100
"""

import os
import sys
import argparse
import yaml
import shutil
import datetime
import csv
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_sample_dataset_yaml(output_dir: str) -> str:
    """
    Create a sample dataset YAML for YOLOv8 training.
    
    In practice, you would:
    1. Download a subset of COCO (person class) or VisDrone dataset
    2. Annotate custom surveillance images using LabelImg/Roboflow
    3. Export in YOLO format
    
    This function creates the dataset config structure that YOLOv8 expects.
    """
    dataset_config = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'person'
        },
        'nc': 1  # number of classes
    }
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # Save dataset YAML
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"[Dataset] Config saved to: {yaml_path}")
    print(f"[Dataset] Place your YOLO-format images and labels in:")
    print(f"  Images: {output_dir}/images/[train|val|test]/")
    print(f"  Labels: {output_dir}/labels/[train|val|test]/")
    
    return yaml_path


def fine_tune(
    dataset_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    lr: float = 0.001,
    output_dir: str = "results/yolo_training",
    device: str = "auto"
):
    """
    Fine-tune YOLOv8 on a custom dataset.
    
    Args:
        dataset_yaml: Path to dataset YAML configuration
        base_model: Pre-trained model to start from
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        lr: Initial learning rate
        output_dir: Directory for training outputs
        device: Training device ("cpu", "cuda", "auto")
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    import torch
    
    # Resolve device
    if device == "auto":
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("YOLOv8 Fine-Tuning for Person Detection")
    print("=" * 60)
    print(f"  Base model:    {base_model}")
    print(f"  Dataset:       {dataset_yaml}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Image size:    {img_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Device:        {device}")
    print(f"  Output:        {output_dir}")
    print("=" * 60)
    
    # Load pre-trained model
    model = YOLO(base_model)
    
    # Start training
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=lr,
        device=device,
        project=output_dir,
        name="sentinel_finetune",
        exist_ok=True,
        # Data augmentation settings
        hsv_h=0.015,     # Hue augmentation
        hsv_s=0.7,       # Saturation augmentation
        hsv_v=0.4,       # Value augmentation
        degrees=10.0,    # Rotation augmentation
        translate=0.1,   # Translation augmentation
        scale=0.5,       # Scale augmentation
        fliplr=0.5,      # Horizontal flip probability
        mosaic=1.0,      # Mosaic augmentation
        mixup=0.1,       # MixUp augmentation
        # Optimization
        optimizer='AdamW',
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,     # Cosine learning rate schedule
        # Logging
        plots=True,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        verbose=True,
    )
    
    # ============================
    # Post-Training Evaluation
    # ============================
    print("\n--- Evaluating on Validation Set ---")
    val_results = model.val()
    
    # Extract key metrics
    metrics = {
        'mAP50': float(val_results.box.map50) if hasattr(val_results.box, 'map50') else 0.0,
        'mAP50-95': float(val_results.box.map) if hasattr(val_results.box, 'map') else 0.0,
        'precision': float(val_results.box.mp) if hasattr(val_results.box, 'mp') else 0.0,
        'recall': float(val_results.box.mr) if hasattr(val_results.box, 'mr') else 0.0,
    }
    
    print("\n--- Validation Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save metrics to CSV
    metrics_dir = "results/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = os.path.join(metrics_dir, "yolo_metrics.csv")
    file_exists = os.path.exists(metrics_file)
    
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'model', 'epochs'] + list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        
        row = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': base_model,
            'epochs': epochs,
            **metrics
        }
        writer.writerow(row)
    
    print(f"\n[Saved] Metrics appended to: {metrics_file}")
    
    # Copy best model to models/ directory
    best_model_path = os.path.join(output_dir, "sentinel_finetune", "weights", "best.pt")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    if os.path.exists(best_model_path):
        dest = os.path.join(models_dir, "yolov8n_finetuned.pt")
        shutil.copy2(best_model_path, dest)
        print(f"[Saved] Best model copied to: {dest}")
    
    # ============================
    # Export to ONNX (optional, for edge deployment)
    # ============================
    try:
        onnx_path = model.export(format='onnx', imgsz=img_size, simplify=True)
        print(f"[Exported] ONNX model: {onnx_path}")
    except Exception as e:
        print(f"[Warning] ONNX export failed: {e}")
    
    return metrics


def run_benchmark(model_path: str = "models/yolov8n.pt", num_frames: int = 100):
    """
    Benchmark model inference speed.
    
    Measures FPS on the current hardware for the resume's 
    'achieved Y FPS' claim.
    """
    import time
    import numpy as np
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed.")
        return
    
    model = YOLO(model_path)
    
    print(f"\n--- Benchmarking {model_path} ---")
    print(f"  Running {num_frames} frames...")
    
    # Create dummy frames matching typical webcam resolution
    dummy_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                    for _ in range(num_frames)]
    
    # Warm up
    for _ in range(5):
        model(dummy_frames[0], verbose=False)
    
    # Benchmark
    times = []
    for frame in dummy_frames:
        start = time.time()
        model(frame, verbose=False)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    avg_fps = 1.0 / avg_time
    
    print(f"  Avg inference time: {avg_time*1000:.1f} ms")
    print(f"  Avg FPS: {avg_fps:.1f}")
    print(f"  Min FPS: {1.0/max(times):.1f}")
    print(f"  Max FPS: {1.0/min(times):.1f}")
    
    # Save benchmark results
    metrics_dir = "results/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    benchmark_file = os.path.join(metrics_dir, "inference_benchmark.csv")
    file_exists = os.path.exists(benchmark_file)
    
    with open(benchmark_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'model', 'avg_fps', 'avg_ms', 'num_frames'
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model_path,
            'avg_fps': round(avg_fps, 1),
            'avg_ms': round(avg_time * 1000, 1),
            'num_frames': num_frames
        })
    
    print(f"  [Saved] Benchmark results to: {benchmark_file}")
    
    return {'avg_fps': avg_fps, 'avg_ms': avg_time * 1000}


# ==========================================
# CLI Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for person detection")
    parser.add_argument("--mode", choices=["train", "benchmark", "setup"],
                        default="setup", help="Mode: train, benchmark, or setup dataset")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Project config path")
    
    args = parser.parse_args()
    
    if args.mode == "setup":
        # Create dataset structure
        create_sample_dataset_yaml("data/yolo_dataset")
        print("\n[Next Steps]")
        print("1. Add images to data/yolo_dataset/images/[train|val]/")
        print("2. Add YOLO-format labels to data/yolo_dataset/labels/[train|val]/")
        print("3. Run: python training/fine_tune_yolo.py --mode train --dataset data/yolo_dataset/dataset.yaml")
    
    elif args.mode == "train":
        if args.dataset is None:
            print("Error: --dataset required for training.")
            print("Run with --mode setup first to create the dataset structure.")
            sys.exit(1)
        
        config = load_config(args.config)
        train_config = config['training']['yolo']
        
        fine_tune(
            dataset_yaml=args.dataset,
            base_model=args.model,
            epochs=args.epochs or train_config['epochs'],
            batch_size=args.batch or train_config['batch_size'],
            img_size=train_config['img_size'],
            lr=train_config['learning_rate'],
            device=config['detection'].get('device', 'auto')
        )
    
    elif args.mode == "benchmark":
        run_benchmark(args.model)
