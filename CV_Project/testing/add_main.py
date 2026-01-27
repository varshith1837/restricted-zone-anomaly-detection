import sys
import cv2
import os
import numpy as np
import face_recognition
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget)
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# ==========================================
# GLOBAL VARIABLES
# ==========================================
area_points = []  # Stores the [x, y] of the points
scale_x = 1.0     # Scaling factor for Width
scale_y = 1.0     # Scaling factor for Height

# ==========================================
# WORKER THREAD (Video & Recognition)
# ==========================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_encodings()

        # Optimization variables
        self.process_this_frame = True
        self.face_locations = []
        self.face_names = []

    def load_encodings(self):
        print("--- Loading Database ---")
        dataset_path = "dataset"
        if not os.path.exists(dataset_path):
            print("Error: 'dataset' folder missing.")
            return

        count = 0
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
                                self.known_face_encodings.append(encodings[0])
                                self.known_face_names.append(user_name)
                                count += 1
                        except:
                            pass
        print(f"Database Loaded: {count} faces.")

    def run(self):
        cap = cv2.VideoCapture(0)
        
        # Alert Cooldown (Wait 5 seconds between screenshots)
        last_screenshot_time = datetime.datetime.now() - datetime.timedelta(seconds=10)

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break

            # Update global scaling factors 
            global scale_x, scale_y
            h, w, _ = frame.shape
            scale_x = w / 800
            scale_y = h / 600

            # =================================================================
            # 1. PRE-PROCESSING (Lag Fix)
            # Scale 0.5 is a good balance between speed and distance
            # =================================================================
            resize_val = 0.50
            small_frame = cv2.resize(frame, (0, 0), fx=resize_val, fy=resize_val)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            rgb_small = np.ascontiguousarray(rgb_small)

            # Only process every other frame to fix lag
            if self.process_this_frame:
                
                # =============================================================
                # 2. FACE DETECTION (Lag Fix)
                # Removed 'number_of_times_to_upsample=2' -> Set to default (1)
                # This drastically reduces CPU usage/Lag
                # =============================================================
                self.face_locations = face_recognition.face_locations(rgb_small)
                
                try:
                    face_encodings = face_recognition.face_encodings(rgb_small, self.face_locations)
                except:
                    face_encodings = []

                self.face_names = []
                for face_encoding in face_encodings:
                    # =========================================================
                    # 3. RECOGNITION LOGIC (False Positive Fix)
                    # =========================================================
                    
                    # Calculate distance to all known faces
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    name = "Unknown"
                    
                    if len(face_distances) > 0:
                        # Find the best match (smallest distance)
                        best_match_index = np.argmin(face_distances)
                        best_score = face_distances[best_match_index]

                        # STRICTER THRESHOLD:
                        # 0.6 is default tolerance. 
                        # We use 0.50 as a strict cutoff to prevent strangers from matching.
                        # If score is < 0.50, it's definitely the person.
                        # If score is > 0.50, it's likely a stranger or too ambiguous.
                        if best_score < 0.50:
                            name = self.known_face_names[best_match_index]
                        else:
                            name = "Unknown"

                    self.face_names.append(name)

            self.process_this_frame = not self.process_this_frame

            # 4. DRAWING & ZONES
            for pt in area_points:
                cv2.circle(frame, (pt[0], pt[1]), 8, (0, 0, 255), -1)

            if len(area_points) > 1:
                pts_array = np.array(area_points, np.int32).reshape((-1, 1, 2))
                is_closed = (len(area_points) == 4)
                cv2.polylines(frame, [pts_array], is_closed, (0, 255, 255), 2)

            zone_active = (len(area_points) == 4)
            if zone_active:
                overlay = frame.copy()
                pts_array = np.array(area_points, np.int32)
                cv2.fillPoly(overlay, [pts_array], (0, 255, 255))
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            intruder_alert = False

            # 5. DRAW FACES
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Restore coordinates (multiply by 2 since we scaled by 0.5)
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

                color = (0, 255, 0) # Green
                
                if name == "Unknown":
                    if zone_active and is_inside:
                        color = (0, 0, 255) # Red
                        intruder_alert = True
                    else:
                        color = (0, 165, 255) # Orange
                
                # Draw Box & Name
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Name background
                text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (left, bottom), (left + text_size[0] + 10, bottom + 30), color, -1)
                cv2.putText(frame, name, (left + 5, bottom + 22), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

            # 6. ALARM LOGIC
            if intruder_alert:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.putText(frame, "RESTRICTED AREA!", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                current_time = datetime.datetime.now()
                if (current_time - last_screenshot_time).total_seconds() > 5:
                    last_screenshot_time = current_time
                    if not os.path.exists("intruders"):
                        os.makedirs("intruders")
                    filename = f"intruders/Intruder_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot Saved: {filename}")

            # 7. PERSON COUNT (Always Visible)
            person_count = len(self.face_locations)
            cv2.rectangle(frame, (0, 0), (230, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"Person Count: {person_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            self.change_pixmap_signal.emit(frame)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.quit()
        self.wait()

# ==========================================
# MAIN WINDOW
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Security System (Click 4 points for Area)")
        self.setFixedSize(800, 600)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(0, 0, 800, 600)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.image_label.setCursor(Qt.CrossCursor)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if len(area_points) < 4:
                x = event.x()
                y = event.y()
                real_x = int(x * scale_x)
                real_y = int(y * scale_y)
                area_points.append([real_x, real_y])
                print(f"Point Added: {real_x}, {real_y}")
            else:
                print("Zone already full. Right click to reset.")
        elif event.button() == Qt.RightButton:
            area_points.clear()
            print("Zone Cleared.")

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        rgb_image = np.ascontiguousarray(rgb_image)
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.IgnoreAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())