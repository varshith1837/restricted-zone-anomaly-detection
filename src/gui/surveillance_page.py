"""
Sentinel Vision — Surveillance Page
=====================================
PyQt5 widget for real-time AI-powered surveillance.
Runs a lightweight pipeline that won't freeze the GUI.
"""

import cv2
import os
import sys
import time
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt


class SurveillanceThread(QThread):
    """
    Camera thread with ML detection.
    
    Loads only what's available — won't crash if models aren't trained yet.
    Uses QImage signals instead of raw numpy to keep GUI responsive.
    """
    change_pixmap_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.zone_points = []

    def add_zone_point(self, x, y):
        if len(self.zone_points) < 4:
            self.zone_points.append([x, y])

    def clear_zone(self):
        self.zone_points.clear()

    def run(self):
        # ---- Load models one by one (skip if unavailable) ----
        person_detector = None
        face_classifier_loaded = False
        known_encodings = []
        known_names = []

        # 1. Try loading YOLOv8
        try:
            from ultralytics import YOLO
            model_path = "models/yolov8n.pt"
            if os.path.exists(model_path):
                person_detector = YOLO(model_path)
                self.status_signal.emit("YOLOv8 loaded ✓")
                print("[OK] YOLOv8 loaded")
            else:
                self.status_signal.emit("YOLOv8 not found — skipping")
                print("[Skip] YOLOv8 model not found")
        except Exception as e:
            print(f"[Skip] YOLOv8: {e}")

        # 2. Try loading face database (simple approach using face_recognition)
        try:
            import face_recognition
            dataset_path = "data/faces"
            if os.path.exists(dataset_path):
                for person_name in os.listdir(dataset_path):
                    person_dir = os.path.join(dataset_path, person_name)
                    if not os.path.isdir(person_dir):
                        continue
                    for fname in os.listdir(person_dir):
                        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                            try:
                                img = face_recognition.load_image_file(
                                    os.path.join(person_dir, fname))
                                encs = face_recognition.face_encodings(img)
                                if encs:
                                    known_encodings.append(encs[0])
                                    known_names.append(person_name)
                            except:
                                pass
                if known_encodings:
                    face_classifier_loaded = True
                    self.status_signal.emit(f"Loaded {len(known_encodings)} face(s) ✓")
                    print(f"[OK] Loaded {len(known_encodings)} face encodings")
        except Exception as e:
            print(f"[Skip] Face recognition: {e}")

        # 3. Try loading trained SVM classifier
        svm_classifier = None
        face_embedder = None
        try:
            from src.recognition.face_classifier import FaceClassifier
            from src.recognition.face_embedder import FaceEmbedder
            classifier = FaceClassifier()
            if classifier.load():
                svm_classifier = classifier
                face_embedder = FaceEmbedder()
                self.status_signal.emit("SVM face classifier loaded ✓")
                print("[OK] SVM classifier loaded")
        except Exception as e:
            print(f"[Skip] SVM classifier: {e}")

        self.status_signal.emit("Monitoring...")

        # ---- Open camera ----
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_signal.emit("ERROR: Cannot open camera")
            return

        frame_count = 0
        face_locations = []
        face_names = []
        process_this_frame = True

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_count += 1
            h, w, _ = frame.shape

            # ---- Person detection (YOLOv8) every 3rd frame ----
            person_boxes = []
            if person_detector and frame_count % 3 == 0:
                try:
                    results = person_detector(frame, conf=0.5, classes=[0], verbose=False)
                    for r in results:
                        if r.boxes:
                            for box in r.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                conf = float(box.conf[0])
                                person_boxes.append((x1, y1, x2, y2, conf))
                except:
                    pass

            # ---- Face recognition every 2nd frame ----
            if face_classifier_loaded and process_this_frame:
                try:
                    import face_recognition
                    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                    face_locations = face_recognition.face_locations(rgb_small)
                    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

                    face_names = []
                    for enc in face_encodings:
                        name = "Unknown"
                        if svm_classifier and face_embedder:
                            # Use trained SVM
                            emb = face_embedder.extract_embedding(
                                cv2.resize(small, (160, 160)))
                            if emb is not None:
                                name, _ = svm_classifier.predict(emb)
                        else:
                            # Fallback: distance matching
                            dists = face_recognition.face_distance(known_encodings, enc)
                            if len(dists) > 0:
                                best = np.argmin(dists)
                                if dists[best] < 0.50:
                                    name = known_names[best]
                        face_names.append(name)
                except Exception as e:
                    print(f"[Face error] {e}")

            process_this_frame = not process_this_frame

            # ---- Draw zone ----
            for pt in self.zone_points:
                cv2.circle(frame, (pt[0], pt[1]), 8, (0, 0, 255), -1)
            if len(self.zone_points) > 1:
                pts = np.array(self.zone_points, np.int32).reshape((-1, 1, 2))
                closed = len(self.zone_points) == 4
                cv2.polylines(frame, [pts], closed, (0, 255, 255), 2)
            zone_active = len(self.zone_points) == 4
            if zone_active:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [np.array(self.zone_points, np.int32)], (0, 255, 255))
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # ---- Draw person boxes (YOLOv8) ----
            for (x1, y1, x2, y2, conf) in person_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(frame, f"Person {conf:.0%}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            # ---- Draw face boxes ----
            intruder_alert = False
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 2; right *= 2; bottom *= 2; left *= 2
                cx, cy = (left + right) // 2, (top + bottom) // 2

                is_inside = False
                if zone_active:
                    result = cv2.pointPolygonTest(
                        np.array(self.zone_points, np.int32), (cx, cy), False)
                    is_inside = result >= 0

                if name == "Unknown" and is_inside:
                    color = (0, 0, 255)
                    intruder_alert = True
                elif name == "Unknown":
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom), (left + 90, bottom + 28), color, -1)
                cv2.putText(frame, name, (left + 5, bottom + 20),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # ---- Alert ----
            if intruder_alert:
                cv2.putText(frame, "! RESTRICTED AREA !", (50, 50),
                           cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            # ---- HUD ----
            person_count = max(len(person_boxes), len(face_locations))
            cv2.rectangle(frame, (0, 0), (280, 35), (0, 0, 0), -1)
            cv2.putText(frame, f"Persons: {person_count} | YOLOv8 + FaceNet",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # ---- Emit QImage (thread-safe) ----
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.change_pixmap_signal.emit(qt_img)

            # ---- Frame limiter (~15 FPS to keep things smooth) ----
            time.sleep(0.066)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class SurveillancePage(QWidget):
    """Surveillance UI with zone selection and real-time stats."""

    def __init__(self):
        super().__init__()
        self.thread = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        header = QHBoxLayout()
        title = QLabel("ACTIVE SURVEILLANCE SYSTEM")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ff4d4d;")

        self.status_lbl = QLabel("Status: Inactive")
        self.status_lbl.setStyleSheet("color: #888; font-size: 14px;")

        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.status_lbl)
        layout.addLayout(header)

        instr = QLabel("Left Click: Add Zone Point | Right Click: Clear Zone")
        instr.setStyleSheet("color: #aaa; font-style: italic;")
        instr.setAlignment(Qt.AlignCenter)
        layout.addWidget(instr)

        self.image_label = QLabel("Click 'Surveillance' to start")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ff4d4d; background-color: #000; color: #666; font-size: 16px;")
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setCursor(Qt.CrossCursor)
        self.image_label.mousePressEvent = self.handle_mouse_click
        layout.addWidget(self.image_label)

        self.setLayout(layout)

    def start_system(self):
        if self.thread is None or not self.thread.isRunning():
            self.thread = SurveillanceThread()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.status_signal.connect(self.update_status)
            self.thread.start()
            self.status_lbl.setText("Status: Starting...")
            self.status_lbl.setStyleSheet("color: #ffaa00; font-size: 14px;")

    def stop_system(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("System Paused")
            self.status_lbl.setText("Status: Paused")
            self.status_lbl.setStyleSheet("color: #ffaa00; font-size: 14px;")

    def handle_mouse_click(self, event):
        if self.thread:
            display_w = self.image_label.width()
            display_h = self.image_label.height()
            scale_x = 640 / display_w
            scale_y = 480 / display_h
            x = int(event.pos().x() * scale_x)
            y = int(event.pos().y() * scale_y)

            if event.button() == Qt.LeftButton:
                self.thread.add_zone_point(x, y)
                print(f"Zone point added: ({x}, {y})")
            elif event.button() == Qt.RightButton:
                self.thread.clear_zone()
                print("Zone cleared.")

    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(
            qt_img.scaled(800, 600, Qt.IgnoreAspectRatio)))

    def update_status(self, text):
        self.status_lbl.setText(f"Status: {text}")
        self.status_lbl.setStyleSheet("color: #00ff00; font-size: 14px; font-weight: bold;")
