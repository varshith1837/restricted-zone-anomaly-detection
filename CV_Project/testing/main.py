import sys
import cv2
import os
import numpy as np
import face_recognition
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, 
                             QStackedWidget, QFrame, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer

# ==========================================
# GLOBAL VARIABLES (From add_main.py)
# ==========================================
area_points = []
scale_x = 1.0
scale_y = 1.0

# ==========================================
# UTILITY: LOAD ENCODINGS (Shared)
# ==========================================
def load_known_faces():
    known_encodings = []
    known_names = []
    dataset_path = "dataset"
    
    if not os.path.exists(dataset_path):
        return [], []

    print("--- Loading Database ---")
    for user_name in os.listdir(dataset_path):
        user_folder = os.path.join(dataset_path, user_name)
        if os.path.isdir(user_folder):
            for filename in os.listdir(user_folder):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    try:
                        path = os.path.join(user_folder, filename)
                        image = face_recognition.load_image_file(path)
                        encodings = face_recognition.face_encodings(image)
                        if len(encodings) > 0:
                            known_encodings.append(encodings[0])
                            known_names.append(user_name)
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
    return known_encodings, known_names

# ==========================================
# THREAD 1: REGISTRATION (From add_face.py)
# ==========================================
class RegistrationThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
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
        
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Logic to save frame if requested
            if self.save_next_frame and self.current_name:
                base_dir = "dataset"
                user_path = os.path.join(base_dir, self.current_name)
                if not os.path.exists(user_path):
                    os.makedirs(user_path)
                
                # Determine filename
                existing_files = len(os.listdir(user_path))
                filename = f"{self.current_name}_{existing_files + 1}.jpg"
                file_path = os.path.join(user_path, filename)
                
                cv2.imwrite(file_path, frame)
                print(f"[Saved] {filename}")
                
                # Visual Flash Effect
                flash = np.ones(frame.shape, dtype="uint8") * 255
                frame = cv2.addWeighted(frame, 0.5, flash, 0.5, 0)
                
                self.save_next_frame = False
                self.save_count += 1

            self.change_pixmap_signal.emit(frame)
        
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# ==========================================
# THREAD 2: SURVEILLANCE (From add_main.py)
# ==========================================
class SurveillanceThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        # Load database immediately upon init
        self.known_face_encodings, self.known_face_names = load_known_faces()
        self.process_this_frame = True
        self.face_locations = []
        self.face_names = []

    def run(self):
        global scale_x, scale_y, area_points
        cap = cv2.VideoCapture(0)
        
        last_screenshot_time = datetime.datetime.now() - datetime.timedelta(seconds=10)

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            # Assuming the display area is approx 640x480 for scaling logic in main window
            # But we keep your logic: update global scale based on frame vs widget size
            # NOTE: In this GUI integration, we map clicks differently, 
            # but we keep the backend logic consistent.
            
            # 1. PRE-PROCESSING
            resize_val = 0.50
            small_frame = cv2.resize(frame, (0, 0), fx=resize_val, fy=resize_val)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            rgb_small = np.ascontiguousarray(rgb_small)

            if self.process_this_frame:
                self.face_locations = face_recognition.face_locations(rgb_small)
                try:
                    face_encodings = face_recognition.face_encodings(rgb_small, self.face_locations)
                except:
                    face_encodings = []

                self.face_names = []
                for face_encoding in face_encodings:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        best_score = face_distances[best_match_index]
                        if best_score < 0.50:
                            name = self.known_face_names[best_match_index]
                        else:
                            name = "Unknown"
                    self.face_names.append(name)

            self.process_this_frame = not self.process_this_frame

            # 4. DRAWING ZONES
            # We use the global area_points
            for pt in area_points:
                cv2.circle(frame, (pt[0], pt[1]), 8, (0, 0, 255), -1)

            if len(area_points) > 1:
                pts_array = np.array(area_points, np.int32).reshape((-1, 1, 2))
                is_closed = (len(area_points) == 4)
                cv2.polylines(frame, [pts_array], is_closed, (0, 255, 255), 2)

            zone_active = (len(area_points) == 4)
            intruder_alert = False

            if zone_active:
                overlay = frame.copy()
                pts_array = np.array(area_points, np.int32)
                cv2.fillPoly(overlay, [pts_array], (0, 255, 255))
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            # 5. DRAW FACES
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                multiplier = int(1 / resize_val)
                top *= multiplier
                right *= multiplier
                bottom *= multiplier
                left *= multiplier

                cx = int((left + right) / 2)
                cy = int((top + bottom) / 2)
                
                is_inside = False
                if zone_active:
                    result = cv2.pointPolygonTest(np.array(area_points, np.int32), (cx, cy), False)
                    if result >= 0:
                        is_inside = True

                color = (0, 255, 0) 
                if name == "Unknown":
                    if zone_active and is_inside:
                        color = (0, 0, 255)
                        intruder_alert = True
                    else:
                        color = (0, 165, 255)
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom), (left + 80, bottom + 30), color, -1) # Simple box
                cv2.putText(frame, name, (left + 5, bottom + 22), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

            # 6. ALARM LOGIC
            if intruder_alert:
                cv2.putText(frame, "RESTRICTED AREA!", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                
                current_time = datetime.datetime.now()
                if (current_time - last_screenshot_time).total_seconds() > 5:
                    last_screenshot_time = current_time
                    if not os.path.exists("intruders"):
                        os.makedirs("intruders")
                    filename = f"intruders/Intruder_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Intruder detected! Saved: {filename}")

            # 7. PERSON COUNT
            person_count = len(self.face_locations)
            cv2.rectangle(frame, (0, 0), (230, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"Person Count: {person_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            self.change_pixmap_signal.emit(frame)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# ==========================================
# WIDGET 1: REGISTRATION PAGE
# ==========================================
class RegistrationPage(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        # Header
        title = QLabel("NEW USER REGISTRATION")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00adb5;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Input Form
        form_layout = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter Person's Name...")
        self.name_input.setStyleSheet("padding: 10px; font-size: 16px; border-radius: 5px; border: 1px solid #555; color: white; background-color: #333;")
        form_layout.addWidget(self.name_input)
        
        self.btn_capture = QPushButton("CAPTURE PHOTO (Save)")
        self.btn_capture.setStyleSheet("background-color: #e0a800; color: black; font-weight: bold; padding: 10px; font-size: 14px;")
        self.btn_capture.clicked.connect(self.capture_image)
        form_layout.addWidget(self.btn_capture)
        layout.addLayout(form_layout)

        # Video Feed
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

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

# ==========================================
# WIDGET 2: SURVEILLANCE PAGE
# ==========================================
class SurveillancePage(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("ACTIVE SURVEILLANCE SYSTEM")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ff4d4d;")
        
        self.status_lbl = QLabel("Status: Inactive")
        self.status_lbl.setStyleSheet("color: #888; font-size: 14px;")
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.status_lbl)
        layout.addLayout(header_layout)

        # Instructions
        instr = QLabel("Left Click: Add Zone Point | Right Click: Clear Zone")
        instr.setStyleSheet("color: #aaa; font-style: italic; margin-bottom: 5px;")
        instr.setAlignment(Qt.AlignCenter)
        layout.addWidget(instr)

        # Video Feed
        self.image_label = QLabel("Initializing...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ff4d4d; background-color: #000;")
        self.image_label.setMinimumSize(800, 600)
        # Mouse tracking for zone selection
        self.image_label.setMouseTracking(True)
        self.image_label.setCursor(Qt.CrossCursor)
        self.image_label.mousePressEvent = self.handle_mouse_click
        
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def start_system(self):
        global area_points
        if self.thread is None or not self.thread.isRunning():
            self.thread = SurveillanceThread()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()
            self.status_lbl.setText("Status: Monitoring")
            self.status_lbl.setStyleSheet("color: #00ff00; font-size: 14px; font-weight: bold;")

    def stop_system(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("System Paused")
            self.status_lbl.setText("Status: Paused")
            self.status_lbl.setStyleSheet("color: #ffaa00; font-size: 14px;")

    def handle_mouse_click(self, event):
        global area_points, scale_x, scale_y
        
        # Mapping logic
        # The logic in add_main.py relies on scale_x/y calculated from frame vs window size
        # Since the label resizes images, we need to ensure coordinates map correctly.
        # For simplicity in this integration, we will approximate based on the fixed Label size 
        # vs standard Camera resolution (640x480 usually).
        
        # We assume the displayed image fills the label (800x600)
        display_w = self.image_label.width()
        display_h = self.image_label.height()
        
        # NOTE: The Thread updates scale_x/scale_y based on actual frame size.
        # We need to inverse map the click:
        # Click (x,y) -> Camera (x,y)
        
        # However, your original code updated scale_x = w / 800 inside the thread.
        # This means 800 is the reference.
        
        if event.button() == Qt.LeftButton:
            if len(area_points) < 4:
                x = event.pos().x()
                y = event.pos().y()
                
                # Ensure we use the global scale factors updated by the thread
                # If thread hasn't run yet, defaults are 1.0
                real_x = int(x * scale_x)
                real_y = int(y * scale_y)
                
                area_points.append([real_x, real_y])
                print(f"Point Added: {real_x}, {real_y}")
            else:
                print("Zone Full. Right click to clear.")
                
        elif event.button() == Qt.RightButton:
            area_points.clear()
            print("Zone Cleared.")

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        # Update globals for mouse clicking mapping
        global scale_x, scale_y
        h, w, ch = cv_img.shape
        
        # We are forcing the display to be 800x600 in the label
        target_w = 800
        target_h = 600
        
        scale_x = w / target_w
        scale_y = h / target_h
        
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(target_w, target_h, Qt.IgnoreAspectRatio)
        return QPixmap.fromImage(p)

# ==========================================
# MAIN DASHBOARD WINDOW
# ==========================================
class MainDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Security & Registration System")
        self.setGeometry(100, 100, 1100, 750)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SIDEBAR ---
        sidebar = QFrame()
        sidebar.setFixedWidth(250)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                border-right: 1px solid #444;
            }
            QPushButton {
                background-color: transparent;
                color: #bbb;
                border: none;
                padding: 15px;
                text-align: left;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #333;
                color: white;
            }
            QPushButton:checked {
                background-color: #00adb5;
                color: white;
                font-weight: bold;
            }
        """)
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # App Title
        lbl_title = QLabel("SENTINEL\nVISION")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("padding: 30px; font-size: 22px; font-weight: bold; color: #00adb5;")
        sidebar_layout.addWidget(lbl_title)

        # Nav Buttons
        self.btn_register = QPushButton("  👤  Register Face")
        self.btn_register.setCheckable(True)
        self.btn_register.clicked.connect(lambda: self.switch_page(0))
        
        self.btn_surveillance = QPushButton("  📹  Surveillance")
        self.btn_surveillance.setCheckable(True)
        self.btn_surveillance.clicked.connect(lambda: self.switch_page(1))
        
        self.btn_exit = QPushButton("  ❌  Exit System")
        self.btn_exit.clicked.connect(self.close)

        sidebar_layout.addWidget(self.btn_register)
        sidebar_layout.addWidget(self.btn_surveillance)
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(self.btn_exit)

        # --- CONTENT AREA ---
        self.stacked_widget = QStackedWidget()
        
        # Initialize Pages
        self.page_register = RegistrationPage()
        self.page_surveillance = SurveillancePage()
        
        self.stacked_widget.addWidget(self.page_register)
        self.stacked_widget.addWidget(self.page_surveillance)

        # Add to Main Layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stacked_widget)

        # Default Page
        self.btn_register.setChecked(True)
        self.switch_page(0)

    def switch_page(self, index):
        # Logic to start/stop cameras to prevent conflicts
        if index == 0: # Register Mode
            self.page_surveillance.stop_system()
            self.page_register.start_camera()
            self.btn_register.setChecked(True)
            self.btn_surveillance.setChecked(False)
        elif index == 1: # Surveillance Mode
            self.page_register.stop_camera()
            self.page_surveillance.start_system()
            self.btn_register.setChecked(False)
            self.btn_surveillance.setChecked(True)
            
        self.stacked_widget.setCurrentIndex(index)

    def closeEvent(self, event):
        self.page_register.stop_camera()
        self.page_surveillance.stop_system()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Global Font Setup
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainDashboard()
    window.show()
    sys.exit(app.exec_())