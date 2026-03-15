"""
Sentinel Vision — Registration Page
=====================================
PyQt5 widget for registering new face identities.
"""

import cv2
import os
import time
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QLineEdit, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt


class RegistrationThread(QThread):
    """Camera thread for face registration."""
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.save_next_frame = False
        self.current_name = ""
        self.save_count = 0

    def set_save_request(self, name):
        self.current_name = name
        self.save_next_frame = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Error] Cannot open camera")
            return

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            if self.save_next_frame and self.current_name:
                base_dir = "data/faces"
                user_path = os.path.join(base_dir, self.current_name)
                os.makedirs(user_path, exist_ok=True)
                existing = len(os.listdir(user_path))
                filename = f"{self.current_name}_{existing + 1}.jpg"
                cv2.imwrite(os.path.join(user_path, filename), frame)
                print(f"[Saved] {filename}")
                flash = np.ones(frame.shape, dtype="uint8") * 255
                frame = cv2.addWeighted(frame, 0.5, flash, 0.5, 0)
                self.save_next_frame = False
                self.save_count += 1

            # Convert to QImage in the thread (not the GUI thread)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.change_pixmap_signal.emit(qt_img)

            # Frame rate limiter (~30 FPS) — prevents flooding the GUI
            time.sleep(0.033)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class RegistrationPage(QWidget):
    """Face registration UI page."""

    def __init__(self):
        super().__init__()
        self.thread = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        title = QLabel("NEW USER REGISTRATION")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00adb5;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        form = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter Person's Name...")
        self.name_input.setStyleSheet(
            "padding: 10px; font-size: 16px; border-radius: 5px; "
            "border: 1px solid #555; color: white; background-color: #333;"
        )
        form.addWidget(self.name_input)

        self.btn_capture = QPushButton("CAPTURE PHOTO")
        self.btn_capture.setStyleSheet(
            "background-color: #e0a800; color: black; font-weight: bold; "
            "padding: 10px; font-size: 14px; border-radius: 5px;"
        )
        self.btn_capture.clicked.connect(self.capture_image)
        form.addWidget(self.btn_capture)
        layout.addLayout(form)

        self.image_label = QLabel("Camera Feed Off")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #555; background-color: #111;")
        self.image_label.setMinimumSize(640, 480)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

    def start_camera(self):
        if self.thread is None or not self.thread.isRunning():
            self.thread = RegistrationThread()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()

    def stop_camera(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Camera Stopped")

    def capture_image(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Input Error", "Please enter a name first.")
            return
        if self.thread:
            self.thread.set_save_request(name)

    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img.scaled(640, 480, Qt.KeepAspectRatio)))
