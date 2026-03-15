"""
Sentinel Vision — Face Classifier Training Script
===================================================
Trains SVM and KNN classifiers on FaceNet embeddings.

This script:
1. Loads all registered face images
2. Extracts 512-d FaceNet embeddings
3. Applies data augmentation to increase training samples
4. Splits into train/val sets (stratified)
5. Trains SVM + KNN, compares both
6. Saves best model, classification report, confusion matrix, and ROC curves

Usage:
    python training/train_face_classifier.py
    python training/train_face_classifier.py --augment --compare
"""

import os
import sys
import argparse
import numpy as np
import cv2
import yaml
import json
import datetime
from collections import Counter
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recognition.face_embedder import FaceEmbedder
from src.recognition.face_classifier import FaceClassifier


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def augment_face(face_img: np.ndarray) -> List[np.ndarray]:
    """
    Apply data augmentation to a face image.
    
    Augmentations:
    - Horizontal flip
    - Brightness adjustment (+/-)
    - Slight rotation (-15° to +15°)
    - Gaussian blur (slight)
    
    Args:
        face_img: BGR face image
        
    Returns:
        List of augmented face images
    """
    augmented = []
    h, w = face_img.shape[:2]
    
    # 1. Horizontal flip
    augmented.append(cv2.flip(face_img, 1))
    
    # 2. Brightness increase
    bright = cv2.convertScaleAbs(face_img, alpha=1.2, beta=30)
    augmented.append(bright)
    
    # 3. Brightness decrease
    dark = cv2.convertScaleAbs(face_img, alpha=0.8, beta=-30)
    augmented.append(dark)
    
    # 4. Slight rotation (+10°)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 10, 1.0)
    rotated = cv2.warpAffine(face_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    augmented.append(rotated)
    
    # 5. Slight rotation (-10°)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -10, 1.0)
    rotated = cv2.warpAffine(face_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    augmented.append(rotated)
    
    return augmented


def load_face_images(dataset_path: str, augment: bool = True) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all face images from the dataset directory.
    
    Args:
        dataset_path: Path to face dataset (data/faces/)
        augment: Whether to apply data augmentation
        
    Returns:
        (images, labels) — lists of face images and person names
    """
    images = []
    labels = []
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return [], []
    
    print(f"\n--- Loading Face Dataset from: {dataset_path} ---")
    
    for person_name in sorted(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        person_images = 0
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                filepath = os.path.join(person_dir, filename)
                img = cv2.imread(filepath)
                
                if img is None:
                    continue
                
                images.append(img)
                labels.append(person_name)
                person_images += 1
                
                # Apply augmentation
                if augment:
                    for aug_img in augment_face(img):
                        images.append(aug_img)
                        labels.append(person_name)
        
        total = person_images * (6 if augment else 1)  # 1 original + 5 augmented
        print(f"  {person_name}: {person_images} originals → {total} total")
    
    print(f"\n  Total images: {len(images)}")
    print(f"  Total classes: {len(set(labels))}")
    print(f"  Class distribution: {dict(Counter(labels))}")
    
    return images, labels


def extract_all_embeddings(embedder: FaceEmbedder, images: List[np.ndarray],
                            batch_size: int = 32) -> np.ndarray:
    """
    Extract embeddings for all face images.
    
    Args:
        embedder: FaceEmbedder instance
        images: List of face images
        batch_size: Batch size for extraction
        
    Returns:
        (N, 512) numpy array of embeddings
    """
    print(f"\n--- Extracting Embeddings ({len(images)} images) ---")
    
    all_embeddings = []
    valid_indices = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        embeddings = embedder.extract_batch_embeddings(batch)
        
        for j, emb in enumerate(embeddings):
            if emb is not None:
                all_embeddings.append(emb)
                valid_indices.append(i + j)
        
        progress = min(i + batch_size, len(images))
        print(f"  Processed {progress}/{len(images)} images...")
    
    print(f"  Valid embeddings: {len(all_embeddings)} / {len(images)}")
    
    return np.array(all_embeddings), valid_indices


def plot_results(eval_results: dict, save_dir: str):
    """
    Generate and save evaluation plots.
    
    Plots:
    - Confusion matrix heatmap
    - Per-class F1 scores bar chart
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[Warning] matplotlib/seaborn not installed. Skipping plots.")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = np.array(eval_results['confusion_matrix'])
    classes = eval_results['classes']
    
    fig, ax = plt.subplots(figsize=(max(8, len(classes)), max(6, len(classes) * 0.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Face Classification — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'face_confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"  [Saved] Confusion matrix plot")
    
    # 2. Per-class F1 scores
    report = eval_results['classification_report']
    class_names = [c for c in classes if c in report]
    f1_scores = [report[c]['f1-score'] for c in class_names]
    
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8), 5))
    bars = ax.bar(range(len(class_names)), f1_scores, color='steelblue')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_title('Face Classification — Per-Class F1 Scores')
    ax.set_ylim(0, 1.1)
    
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'face_f1_scores.png'), dpi=150)
    plt.close()
    print(f"  [Saved] F1 scores plot")


def train_and_evaluate(config_path: str = "configs/config.yaml",
                        augment: bool = True,
                        compare_models: bool = True):
    """
    Main training pipeline.
    
    1. Load images → 2. Extract embeddings → 3. Split → 4. Train → 5. Evaluate
    """
    config = load_config(config_path)
    dataset_path = config['paths']['face_dataset']
    
    # ============================
    # 1. Load face images
    # ============================
    images, labels = load_face_images(dataset_path, augment=augment)
    
    if len(images) == 0:
        print("\nError: No face images found!")
        print(f"Please add face images to: {dataset_path}/[person_name]/[image.jpg]")
        return
    
    if len(set(labels)) < 2:
        print("\nError: Need at least 2 different people for classification!")
        return
    
    # ============================
    # 2. Extract embeddings
    # ============================
    embedder = FaceEmbedder(config_path)
    embeddings, valid_indices = extract_all_embeddings(embedder, images)
    
    # Filter labels to match valid embeddings
    valid_labels = np.array([labels[i] for i in valid_indices])
    
    # ============================
    # 3. Train/Val split (stratified)
    # ============================
    from sklearn.model_selection import train_test_split
    
    test_size = config['training']['face_classifier']['test_size']
    
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, valid_labels,
        test_size=test_size,
        stratify=valid_labels,
        random_state=42
    )
    
    print(f"\n--- Data Split ---")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    
    # ============================
    # 4. Train classifier(s)
    # ============================
    results = {}
    
    if compare_models:
        # Train both SVM and KNN, compare
        for clf_type in ['svm', 'knn']:
            print(f"\n{'='*60}")
            print(f"Training {clf_type.upper()} classifier...")
            print(f"{'='*60}")
            
            classifier = FaceClassifier(config_path)
            classifier.classifier_type = clf_type
            
            train_metrics = classifier.train(X_train, y_train, do_grid_search=True)
            eval_metrics = classifier.evaluate(X_val, y_val)
            
            results[clf_type] = {
                'train_metrics': train_metrics,
                'eval_metrics': eval_metrics,
                'classifier': classifier
            }
    else:
        # Train only the configured classifier
        classifier = FaceClassifier(config_path)
        train_metrics = classifier.train(X_train, y_train, do_grid_search=True)
        eval_metrics = classifier.evaluate(X_val, y_val)
        
        results[classifier.classifier_type] = {
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'classifier': classifier
        }
    
    # ============================
    # 5. Compare & save best
    # ============================
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    best_name = None
    best_acc = 0.0
    
    for name, res in results.items():
        acc = res['eval_metrics']['accuracy']
        cv_mean = res['train_metrics'].get('cv_mean', 0.0)
        print(f"  {name.upper()}: Val Accuracy = {acc:.4f}, CV Mean = {cv_mean:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_name = name
    
    print(f"\n  >>> Best Model: {best_name.upper()} (accuracy: {best_acc:.4f})")
    
    # Save best model
    best_classifier = results[best_name]['classifier']
    best_classifier.save()
    
    # ============================
    # 6. Generate plots & save metrics
    # ============================
    plots_dir = config['paths']['plots_dir']
    metrics_dir = config['paths']['metrics_dir']
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Plot results for best model
    plot_results(results[best_name]['eval_metrics'], plots_dir)
    
    # Save metrics to JSON
    metrics_file = os.path.join(metrics_dir, "face_classifier_metrics.json")
    save_metrics = {}
    for name, res in results.items():
        save_metrics[name] = {
            'train_metrics': {k: v for k, v in res['train_metrics'].items()},
            'eval_accuracy': res['eval_metrics']['accuracy'],
            'eval_report': res['eval_metrics']['classification_report']
        }
    save_metrics['best_model'] = best_name
    save_metrics['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(metrics_file, 'w') as f:
        json.dump(save_metrics, f, indent=2, default=str)
    
    print(f"\n[Saved] Metrics to: {metrics_file}")
    print(f"[Saved] Plots to: {plots_dir}")
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")


# ==========================================
# CLI Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face classifier on embeddings")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Config path")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--compare", action="store_true", default=True,
                        help="Compare SVM and KNN (default: True)")
    parser.add_argument("--no-compare", action="store_true",
                        help="Train only the configured classifier type")
    
    args = parser.parse_args()
    
    train_and_evaluate(
        config_path=args.config,
        augment=not args.no_augment,
        compare_models=not args.no_compare
    )
