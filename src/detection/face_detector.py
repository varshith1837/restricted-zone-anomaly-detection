"""
Sentinel Vision — MTCNN Face Detector
======================================
Uses MTCNN (Multi-task Cascaded Convolutional Networks) for face detection.

Key ML concepts demonstrated:
- Cascaded CNN architecture (P-Net → R-Net → O-Net)
- Multi-task learning (detection + landmark localization)
- Confidence-based face filtering
"""

import cv2
import numpy as np
import yaml
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FaceDetection:
    """Single face detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5 facial landmarks
    face_crop: Optional[np.ndarray] = None  # Cropped and aligned face image
    
    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        self.width = x2 - x1
        self.height = y2 - y1


class FaceDetector:
    """
    MTCNN-based face detector.
    
    Replaces face_recognition.face_locations() with a more robust
    deep learning detector that provides:
    - Higher accuracy in varied conditions
    - Facial landmarks (eyes, nose, mouth) for alignment
    - Confidence scores per detection
    
    The detector feeds cropped faces into the FaceEmbedder for recognition.
    
    Args:
        config_path: Path to config.yaml
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        face_config = config['face_detection']
        
        self.min_face_size = face_config['min_face_size']
        self.thresholds = face_config['thresholds']
        
        # Determine device
        device_str = face_config.get('device', 'auto')
        
        try:
            import torch
            from facenet_pytorch import MTCNN
        except ImportError:
            raise ImportError(
                "facenet-pytorch not installed. Run: pip install facenet-pytorch"
            )
        
        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)
        
        # Initialize MTCNN
        self.mtcnn = MTCNN(
            image_size=160,            # Standard FaceNet input size
            margin=20,                 # Margin around detected face
            min_face_size=self.min_face_size,
            thresholds=self.thresholds,
            factor=0.709,              # Scale factor for image pyramid
            post_process=True,         # Normalize pixel values
            select_largest=False,      # Detect all faces, not just largest
            keep_all=True,             # Keep all detected faces
            device=self.device
        )
        
        print(f"[FaceDetector] MTCNN loaded on {self.device}")
    
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of FaceDetection objects
        """
        # MTCNN expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces — returns boxes, probs, landmarks
        boxes, probs, landmarks = self.mtcnn.detect(rgb_frame, landmarks=True)
        
        detections = []
        
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob is None:
                    continue
                
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Clamp to frame boundaries
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Skip tiny or invalid boxes
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                
                # Crop face region
                face_crop = frame[y1:y2, x1:x2].copy()
                
                # Get landmarks for this face
                face_landmarks = landmarks[i] if landmarks is not None else None
                
                det = FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(prob),
                    landmarks=face_landmarks,
                    face_crop=face_crop
                )
                detections.append(det)
        
        return detections
    
    def detect_and_draw(self, frame: np.ndarray,
                         color: Tuple[int, int, int] = (255, 255, 0),
                         draw_landmarks: bool = True) -> Tuple[np.ndarray, List[FaceDetection]]:
        """
        Detect faces and draw annotations on frame.
        
        Args:
            frame: BGR image
            color: Box color (BGR)
            draw_landmarks: Whether to draw facial landmarks
            
        Returns:
            Annotated frame, list of detections
        """
        detections = self.detect(frame)
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw face box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence
            label = f"Face {det.confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw landmarks (eyes, nose, mouth corners)
            if draw_landmarks and det.landmarks is not None:
                for point in det.landmarks:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)
        
        return annotated, detections


# ==========================================
# Standalone test
# ==========================================
if __name__ == "__main__":
    import time
    
    print("=" * 50)
    print("FaceDetector — Standalone Test")
    print("=" * 50)
    
    detector = FaceDetector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit()
    
    print("Running face detection... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        annotated, detections = detector.detect_and_draw(frame)
        fps = 1.0 / max(time.time() - start, 1e-6)
        
        cv2.putText(annotated, f"FPS: {fps:.1f} | Faces: {len(detections)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("MTCNN Face Detection", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
