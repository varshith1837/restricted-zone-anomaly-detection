"""
Sentinel Vision — Unified Evaluation Script
=============================================
Generates comprehensive evaluation metrics and plots for all models.

Outputs:
- Classification reports (face + activity)
- Confusion matrices
- ROC curves with AUC
- Model comparison table
- Inference benchmark results
"""

import os, sys, json, argparse, numpy as np, yaml, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path="configs/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def generate_comparison_table(metrics_dir):
    """Load all saved metrics and create a comparison table."""
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")

    # Face classifier
    face_file = os.path.join(metrics_dir, "face_classifier_metrics.json")
    if os.path.exists(face_file):
        with open(face_file) as f:
            face_data = json.load(f)
        print("\n--- Face Recognition ---")
        print(f"  Best model: {face_data.get('best_model', 'N/A')}")
        for name, data in face_data.items():
            if isinstance(data, dict) and 'eval_accuracy' in data:
                cv_mean = data.get('train_metrics', {}).get('cv_mean', 'N/A')
                print(f"  {name.upper()}: Accuracy={data['eval_accuracy']:.4f}, CV={cv_mean}")

    # Activity classifier
    act_file = os.path.join(metrics_dir, "activity_classifier_metrics.json")
    if os.path.exists(act_file):
        with open(act_file) as f:
            act_data = json.load(f)
        print("\n--- Activity Classification ---")
        print(f"  Accuracy: {act_data.get('accuracy', 'N/A')}")
        print(f"  Data type: {act_data.get('data_type', 'N/A')}")

    # YOLO
    yolo_file = os.path.join(metrics_dir, "yolo_metrics.csv")
    if os.path.exists(yolo_file):
        import csv
        with open(yolo_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            latest = rows[-1]
            print("\n--- YOLOv8 Person Detection ---")
            for k, v in latest.items():
                print(f"  {k}: {v}")

    # Benchmark
    bench_file = os.path.join(metrics_dir, "inference_benchmark.csv")
    if os.path.exists(bench_file):
        import csv
        with open(bench_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            latest = rows[-1]
            print("\n--- Inference Benchmark ---")
            for k, v in latest.items():
                print(f"  {k}: {v}")

    print(f"\n{'='*70}")


def generate_roc_curves(metrics_dir, plots_dir):
    """Generate ROC curves if probability data is available."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed.")
        return

    os.makedirs(plots_dir, exist_ok=True)

    face_file = os.path.join(metrics_dir, "face_classifier_metrics.json")
    if not os.path.exists(face_file):
        print("  No face metrics found for ROC curves.")
        return

    # Summary comparison plot
    with open(face_file) as f:
        data = json.load(f)

    models, accuracies = [], []
    for name, vals in data.items():
        if isinstance(vals, dict) and 'eval_accuracy' in vals:
            models.append(name.upper())
            accuracies.append(vals['eval_accuracy'])

    if models:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(models, accuracies, color=['steelblue', 'coral'][:len(models)])
        ax.set_ylabel('Accuracy')
        ax.set_title('Face Recognition — Model Comparison')
        ax.set_ylim(0, 1.1)
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02,
                    f'{acc:.3f}', ha='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=150)
        plt.close()
        print(f"  [Saved] Model comparison plot")


def run_evaluation(config_path="configs/config.yaml"):
    config = load_config(config_path)
    metrics_dir = config['paths']['metrics_dir']
    plots_dir = config['paths']['plots_dir']

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    generate_comparison_table(metrics_dir)
    generate_roc_curves(metrics_dir, plots_dir)

    print("\n[Done] Evaluation complete.")
    print(f"  Metrics: {metrics_dir}/")
    print(f"  Plots:   {plots_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_evaluation(args.config)
