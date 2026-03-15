# Sentinel Vision — AI-Powered Surveillance System

A real-time multi-model surveillance system combining **YOLOv8 person detection**, **FaceNet-based face recognition**, and **pose-based anomaly detection** with an interactive PyQt5 dashboard.

## Architecture

```
Camera Feed
    │
    ├── YOLOv8 ──── Person Detection (bounding boxes + confidence)
    │
    ├── MTCNN ───── Face Detection (cropped faces + landmarks)
    │   └── FaceNet ── 512-d Embeddings
    │       └── SVM/KNN ── Identity Classification
    │
    ├── MediaPipe ── Pose Estimation (33 landmarks)
    │   └── Random Forest / LSTM ── Activity Classification
    │
    └── Zone Monitor ── Threat Scoring (weighted fusion)
            └── Alert Manager ── Screenshots + CSV Logging
```

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Person Detection | YOLOv8 (Ultralytics) | Real-time person localization |
| Face Detection | MTCNN (facenet-pytorch) | Multi-task cascaded face detection |
| Face Recognition | FaceNet + SVM | 512-d embedding + trained classifier |
| Pose Estimation | MediaPipe BlazePose | 33-point body landmark extraction |
| Activity Recognition | Random Forest / LSTM | Pose-feature classification |
| Zone Monitoring | OpenCV | Polygon-based restricted area |
| GUI | PyQt5 | Desktop dashboard with sidebar |
| Training | scikit-learn, PyTorch | Model training + GridSearchCV |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Register faces (add images to data/faces/[name]/)
python src/gui/main_window.py
# → Use the "Register Face" page to capture images

# Train face classifier
python training/train_face_classifier.py

# Train activity classifier
python training/train_activity_classifier.py

# Run evaluation
python training/evaluate.py

# Launch full system
python src/gui/main_window.py
# → Switch to "Surveillance" mode
```

## Project Structure

```
├── configs/config.yaml           # Central configuration
├── requirements.txt
├── src/
│   ├── detection/
│   │   ├── person_detector.py    # YOLOv8 wrapper
│   │   └── face_detector.py      # MTCNN wrapper
│   ├── recognition/
│   │   ├── face_embedder.py      # FaceNet embedding extractor
│   │   └── face_classifier.py    # SVM/KNN on embeddings
│   ├── anomaly/
│   │   ├── pose_estimator.py     # MediaPipe + feature engineering
│   │   ├── activity_classifier.py # RF/LSTM activity model
│   │   └── zone_monitor.py       # Threat scoring
│   ├── pipeline/
│   │   ├── video_pipeline.py     # Master per-frame orchestrator
│   │   └── alert_manager.py      # Alerts + logging
│   └── gui/
│       ├── main_window.py        # Dashboard
│       ├── registration_page.py  # Face registration
│       └── surveillance_page.py  # Surveillance feed
├── training/
│   ├── fine_tune_yolo.py         # YOLOv8 fine-tuning
│   ├── train_face_classifier.py  # SVM/KNN training
│   ├── train_activity_classifier.py # RF/LSTM training
│   └── evaluate.py               # Unified evaluation
├── models/                       # Saved model weights
├── data/faces/                   # Registered face images
├── results/
│   ├── metrics/                  # JSON/CSV metrics
│   └── plots/                    # Confusion matrices, F1 plots
└── notebooks/analysis.ipynb      # EDA + results visualization
```

## ML Models Trained

1. **SVM Face Classifier** — Trained on FaceNet 512-d embeddings with GridSearchCV (RBF kernel, hyperparameter tuning, 5-fold CV)
2. **KNN Face Classifier** — Distance-weighted KNN baseline for comparison
3. **Random Forest Activity Classifier** — Trained on 19 engineered pose features (joint angles, velocities, distances)
4. **YOLOv8 Person Detector** — Fine-tuned from pre-trained nano weights

## Key ML Features

- **Feature Engineering**: 19 handcrafted pose features (joint angles, inter-keypoint distances, temporal velocity, pose stability)
- **Hyperparameter Tuning**: GridSearchCV for SVM (C, gamma, kernel) and Random Forest (n_estimators, max_depth)
- **Cross-Validation**: Stratified K-fold for robust model evaluation
- **Data Augmentation**: Horizontal flip, brightness, rotation for face images
- **Multi-Signal Threat Scoring**: Weighted fusion of identity + activity + zone intrusion
- **Evaluation Metrics**: Precision, recall, F1, confusion matrix, model comparison plots
