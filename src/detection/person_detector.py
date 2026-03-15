"""
Sentinel Vision — YOLOv8 Person Detector
=========================================
Wraps Ultralytics YOLOv8 for real-time person detection.
Supports both pre-trained and fine-tuned weights.

Key ML concepts demonstrated:
- Object detection with anchor-free architecture
- Non-Maximum Suppression (NMS)
- Confidence thresholding
- Class-specific filtering
"""

import cv2
import numpy as np
import yaml
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Detection:
    """Single person detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0  # 0 = person in COCO
    center: Tuple[int, int] = (0, 0)
    
    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        self.width = x2 - x1
        self.height = y2 - y1
        self.area = self.width * self.height


class PersonDetector:
    """
    YOLOv8-based person detector.
    
    This replaces the basic face_recognition.face_locations() approach
    with a proper trained object detection model that can:
    - Detect full bodies (not just faces)
    - Provide confidence scores for each detection
    - Run at real-time speeds with GPU acceleration
    - Be fine-tuned on custom surveillance datasets
    
    Args:
        config_path: Path to config.yaml
        model_path: Override model path (optional)
    """
    
    def __init__(self, config_path: str = "configs/config.yaml", model_path: Optional[str] = None):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        det_config = config['detection']
        
        self.confidence_threshold = det_config['confidence_threshold']
        self.iou_threshold = det_config['iou_threshold']
        self.target_classes = det_config['target_classes']
        self.input_size = det_config['input_size']
        
        # Resolve model path
        _model_path = model_path or det_config['model_path']
        
        # Import ultralytics here to avoid import errors when not installed
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )
        
        # Determine device
        device = det_config.get('device', 'auto')
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load YOLOv8 model
        if not os.path.exists(_model_path):
            print(f"[PersonDetector] Model not found at {_model_path}, downloading yolov8n.pt...")
            _model_path = 'yolov8n.pt'
        
        self.model = YOLO(_model_path)
        self.model.to(self.device)
        print(f"[PersonDetector] Loaded model on {self.device}")
        
        # Detection statistics
        self.total_detections = 0
        self.total_frames = 0
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons in a frame.
        
        Args:
            frame: BGR image (numpy array) from OpenCV
            
        Returns:
            List of Detection objects for persons found
        """
        self.total_frames += 1
        
        # Run YOLOv8 inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            classes=self.target_classes,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                # Extract bounding box (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id
                )
                detections.append(detection)
        
        self.total_detections += len(detections)
        return detections
    
    def detect_and_draw(self, frame: np.ndarray, 
                         color: Tuple[int, int, int] = (0, 255, 0),
                         thickness: int = 2) -> Tuple[np.ndarray, List[Detection]]:
        """
        Detect persons and draw bounding boxes on frame.
        
        Args:
            frame: BGR image
            color: Box color (BGR)
            thickness: Box line thickness
            
        Returns:
            Annotated frame, list of detections
        """
        detections = self.detect(frame)
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence label
            label = f"Person {det.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 5, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw center point
            cv2.circle(annotated, det.center, 4, (0, 0, 255), -1)
        
        return annotated, detections
    
    def get_stats(self) -> dict:
        """Return detection statistics."""
        avg = self.total_detections / max(self.total_frames, 1)
        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": round(avg, 2),
            "device": self.device,
            "confidence_threshold": self.confidence_threshold
        }


# ==========================================
# Standalone test
# ==========================================
if __name__ == "__main__":
    import time
    
    print("=" * 50)
    print("PersonDetector — Standalone Test")
    print("=" * 50)
    
    detector = PersonDetector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit()
    
    fps_list = []
    
    print("Running detection... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        annotated, detections = detector.detect_and_draw(frame)
        elapsed = time.time() - start
        fps = 1.0 / max(elapsed, 1e-6)
        fps_list.append(fps)
        
        # Draw FPS
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Persons: {len(detections)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv8 Person Detection", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print stats
    stats = detector.get_stats()
    print(f"\n--- Stats ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if fps_list:
        print(f"  avg_fps: {np.mean(fps_list):.1f}")
