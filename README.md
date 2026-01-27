# Real-Time CCTV Anomaly Detection for Restricted Security Zones

## Overview
This project implements a real-time computer vision–based security system designed to monitor CCTV feeds, recognize authorized individuals, and detect unauthorized intrusions into restricted zones. The system combines face recognition, interactive region-based access control, and low-latency video processing to operate under real-world deployment constraints.

The solution is built using Python and OpenCV, with a multi-threaded architecture and an interactive graphical interface. It is suitable for small- to medium-scale security monitoring scenarios such as offices, laboratories, server rooms, and controlled operational environments.

---

## Key Features
- Real-time face detection and recognition from live CCTV streams
- Identity-based access control using facial embeddings
- Interactive definition of restricted zones using point-and-click interface
- Intrusion detection for unknown individuals entering restricted areas
- Visual alerts and automated evidence capture
- Multi-threaded pipeline to ensure low latency and responsive UI
- Performance optimizations to balance accuracy and throughput

---

## System Architecture

The system is composed of two independent modules:

### 1. Face Registration Module (`add_face.py`)
This module is responsible for collecting face images of authorized users using a webcam and storing them for recognition.

**Responsibilities**
- Capture face images from live camera input
- Store images using a flat file-based structure
- Encourage pose and angle variation for robust recognition
- Persist data for downstream real-time inference

---

### 2. Real-Time Monitoring Module (`add_main.py`)
This module continuously processes live video streams, performs face recognition, enforces restricted zone rules, and generates alerts.

**Core Components**
- **Worker Thread** for video capture, preprocessing, recognition, and intrusion logic
- **GUI Thread** for rendering frames, handling user interaction, and zone definition

---

## Data Storage Structure

### Face Registration Data

The system uses a **flat storage layout** for registered faces.  
All images are stored directly inside a single directory without user-specific subfolders.

**Design Rationale**
- Simplifies file management and deployment
- Avoids nested directory traversal
- Enables identity extraction directly from filename prefixes
- Easier to back up, move, or version-control metadata

### Intrusion Evidence Storage

Images captured during intrusion events are stored separately to preserve forensic evidence.


Each file contains a full-resolution frame captured at the moment an unauthorized individual entered a restricted zone.

---

## Face Recognition and Identity Mapping

During system initialization:
1. All images in the `dataset/` directory are scanned.
2. A face is detected and encoded into a 128-dimensional embedding.
3. The identity label is extracted from the filename prefix.
4. Encodings and identities are stored in memory for real-time matching.

This approach avoids directory-based identity mapping while maintaining accurate recognition.

---

## Restricted Zone Monitoring

- Users define restricted zones by selecting four points on the live video feed.
- The selected points form a polygonal boundary.
- The system checks whether detected face centroids lie inside the polygon.
- An intrusion is flagged when an **unknown individual** enters the restricted area.

Visual feedback is provided through bounding boxes, overlays, and alert messages.

---

## Alert and Evidence System

When an intrusion is detected:
- The frame is visually highlighted
- Warning text is displayed
- A screenshot is captured and saved automatically
- A cooldown mechanism prevents excessive disk writes

---

## Performance Optimizations

The system applies multiple optimizations to support real-time operation:
- Frame downsampling to reduce computational cost
- Alternating frame processing to balance speed and accuracy
- Multi-threaded execution to prevent UI blocking
- Reduced face detection upsampling for lower CPU usage

Typical observed performance:
- Frame Rate: 15–20 FPS
- Recognition Latency: <100 ms
- Detection Accuracy: ~95% in controlled environments

---

## Technology Stack
- Python
- OpenCV
- face_recognition (dlib-based embeddings)
- PyQt5
- NumPy

---

## How to Run

### Install Dependencies
```bash
pip install opencv-python face_recognition PyQt5 numpy

