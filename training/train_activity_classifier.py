"""
Sentinel Vision — Activity Classifier Training Script
======================================================
Generates synthetic pose data and trains the activity classifier.

Usage:
    python training/train_activity_classifier.py
"""

import os, sys, argparse, numpy as np, json, yaml, datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.anomaly.activity_classifier import ActivityClassifier


def load_config(path="configs/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def generate_synthetic_data(n_per_class=200):
    """Generate synthetic pose features based on biomechanical priors."""
    np.random.seed(42)
    features_all, labels_all = [], []

    profiles = {
        'normal_walking': dict(elbow=(145,20), shoulder=(30,10), knee=(155,15), sw=(0.8,0.1), hw=(0.5,0.08), al=(0.7,0.1), fs=(0.4,0.1), ar=(2.5,0.3), com=(0,0.05), arm_p=0.05, bv=(0.03,0.01), hv=(0.02,0.008), um=(0.02,0.01), st=(0.01,0.005)),
        'running':        dict(elbow=(110,25), shoulder=(60,15), knee=(120,20), sw=(0.85,0.1), hw=(0.55,0.1), al=(0.75,0.1), fs=(0.7,0.15), ar=(2.2,0.4), com=(0,0.08), arm_p=0.1, bv=(0.10,0.03), hv=(0.08,0.025), um=(0.06,0.02), st=(0.04,0.015)),
        'loitering':      dict(elbow=(160,15), shoulder=(15,8),  knee=(170,10), sw=(0.8,0.1), hw=(0.5,0.08), al=(0.65,0.08), fs=(0.3,0.08), ar=(2.8,0.2), com=(0,0.02), arm_p=0.02, bv=(0.005,0.003), hv=(0.003,0.002), um=(0.005,0.003), st=(0.002,0.001)),
        'suspicious':     dict(elbow=(100,30), shoulder=(80,25), knee=(140,20), sw=(0.85,0.12), hw=(0.5,0.1), al=(0.8,0.12), fs=(0.5,0.15), ar=(2.3,0.4), com=(-0.1,0.1), arm_p=0.5, bv=(0.04,0.025), hv=(0.02,0.015), um=(0.06,0.03), st=(0.03,0.02)),
        'falling':        dict(elbow=(90,35), shoulder=(70,30), knee=(100,30), sw=(0.9,0.15), hw=(0.6,0.12), al=(0.7,0.15), fs=(0.8,0.2), ar=(1.2,0.4), com=(0.2,0.15), arm_p=0.4, bv=(0.15,0.05), hv=(0.12,0.04), um=(0.10,0.04), st=(0.08,0.03)),
    }

    for activity, p in profiles.items():
        for _ in range(n_per_class):
            f = np.zeros(19, dtype=np.float32)
            f[0] = np.random.normal(*p['elbow']); f[1] = np.random.normal(*p['elbow'])
            f[2] = np.random.normal(*p['shoulder']); f[3] = np.random.normal(*p['shoulder'])
            f[4] = np.random.normal(*p['knee']); f[5] = np.random.normal(*p['knee'])
            f[6] = np.random.normal(*p['sw']); f[7] = np.random.normal(*p['hw'])
            f[8] = np.random.normal(*p['al']); f[9] = np.random.normal(*p['al'])
            f[10] = np.random.normal(*p['fs'])
            f[11] = np.random.normal(*p['ar']); f[12] = np.random.normal(*p['com'])
            f[13] = float(np.random.random() < p['arm_p'])
            f[14] = float(np.random.random() < p['arm_p'])
            f[15] = max(0, np.random.normal(*p['bv']))
            f[16] = max(0, np.random.normal(*p['hv']))
            f[17] = max(0, np.random.normal(*p['um']))
            f[18] = max(0, np.random.normal(*p['st']))
            f[0] += np.random.normal(0, 5); f[4] += np.random.normal(0, 5)
            features_all.append(f); labels_all.append(activity)

    return np.array(features_all), np.array(labels_all)


def plot_results(eval_res, feat_imp, feat_names, save_dir):
    """Generate confusion matrix and feature importance plots."""
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt; import seaborn as sns
    except ImportError:
        print("[Warning] matplotlib/seaborn not installed."); return

    os.makedirs(save_dir, exist_ok=True)

    # Confusion matrix
    cm = np.array(eval_res['confusion_matrix']); classes = eval_res['classes']
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Activity Classification — Confusion Matrix')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'activity_confusion_matrix.png'), dpi=150); plt.close()

    # Feature importance
    if feat_imp is not None and len(feat_imp) == len(feat_names):
        idx = np.argsort(feat_imp)[::-1]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(feat_names)), feat_imp[idx], color='coral')
        ax.set_xticks(range(len(feat_names)))
        ax.set_xticklabels([feat_names[i] for i in idx], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Importance'); ax.set_title('Feature Importance (Random Forest)')
        plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'activity_feature_importance.png'), dpi=150); plt.close()

    # Per-class F1
    report = eval_res['classification_report']
    cn = [c for c in classes if c in report]; f1s = [report[c]['f1-score'] for c in cn]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(cn)), f1s, color='darkorange')
    ax.set_xticks(range(len(cn))); ax.set_xticklabels(cn, rotation=30, ha='right')
    ax.set_ylabel('F1 Score'); ax.set_title('Per-Class F1 Scores'); ax.set_ylim(0, 1.1)
    for bar, s in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02, f'{s:.2f}', ha='center', fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'activity_f1_scores.png'), dpi=150); plt.close()
    print(f"  [Saved] Plots to: {save_dir}")


def train_and_evaluate(config_path="configs/config.yaml", n_samples=200):
    config = load_config(config_path)

    print("\n--- Generating Synthetic Training Data ---")
    features, labels = generate_synthetic_data(n_per_class=n_samples)
    print(f"  Total: {len(features)}, Classes: {dict(Counter(labels))}")

    from sklearn.model_selection import train_test_split
    test_size = config['training']['activity_classifier']['test_size']
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=test_size, stratify=labels, random_state=42)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    classifier = ActivityClassifier(config_path)
    classifier.train(X_train, y_train, do_grid_search=True)
    eval_results = classifier.evaluate(X_val, y_val)
    classifier.save()

    from src.anomaly.pose_estimator import PoseEstimator
    feat_names = PoseEstimator.get_feature_names(None)
    plot_results(eval_results, classifier.feature_importances_, feat_names, config['paths']['plots_dir'])

    metrics_dir = config['paths']['metrics_dir']; os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "activity_classifier_metrics.json"), 'w') as f:
        json.dump({'timestamp': datetime.datetime.now().isoformat(), 'accuracy': eval_results['accuracy'],
                   'report': eval_results['classification_report'], 'data_type': 'synthetic'}, f, indent=2, default=str)

    print(f"\n{'='*60}\nACTIVITY TRAINING COMPLETE\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    train_and_evaluate(args.config, args.samples)
