"""
Sentinel Vision — Video Pipeline
===================================
Orchestrates all ML modules for per-frame processing.

Pipeline per frame:
1. YOLOv8 → Person detection
2. MTCNN → Face detection (per person crop)
3. FaceNet → Embedding extraction
4. SVM/KNN → Face classification
5. MediaPipe → Pose estimation
6. Random Forest → Activity classification
7. Zone Monitor → Threat scoring
"""

import cv2
import numpy as np
import yaml
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass

from src.detection.person_detector import PersonDetector, Detection
from src.detection.face_detector import FaceDetector, FaceDetection
from src.recognition.face_embedder import FaceEmbedder
from src.recognition.face_classifier import FaceClassifier
from src.anomaly.pose_estimator import PoseEstimator, PoseResult
from src.anomaly.activity_classifier import ActivityClassifier
from src.anomaly.zone_monitor import ZoneMonitor, ThreatAssessment


@dataclass
class FrameResult:
    """Complete analysis result for a single frame."""
    person_detections: List[Detection]
    face_detections: List[FaceDetection]
    pose_results: List[Optional[PoseResult]]
    threat_assessments: List[ThreatAssessment]
    annotated_frame: np.ndarray
    inference_time_ms: float
    person_count: int
    alert_count: int


class VideoPipeline:
    """
    Master pipeline that processes each video frame through all ML modules.
    
    Designed for real-time inference with configurable module enable/disable.
    
    Args:
        config_path: Path to config.yaml
        enable_face: Enable face recognition pipeline
        enable_pose: Enable pose estimation + activity classification
    """
    
    def __init__(self, config_path: str = "configs/config.yaml",
                 enable_face: bool = True, enable_pose: bool = True):
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.enable_face = enable_face
        self.enable_pose = enable_pose
        
        print("=" * 60)
        print("Initializing Sentinel Vision Pipeline")
        print("=" * 60)
        
        # Module 1: Person Detection
        self.person_detector = PersonDetector(config_path)
        
        # Module 2: Face Recognition
        if enable_face:
            self.face_detector = FaceDetector(config_path)
            self.face_embedder = FaceEmbedder(config_path)
            self.face_classifier = FaceClassifier(config_path)
            self.face_classifier.load()
        
        # Module 3: Activity Recognition
        if enable_pose:
            self.pose_estimator = PoseEstimator(config_path)
            self.activity_classifier = ActivityClassifier(config_path)
            self.activity_classifier.load()
        
        # Zone Monitor
        self.zone_monitor = ZoneMonitor(config_path)
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0.0
        self.process_every_n = self.config['camera']['process_every_n_frames']
        
        # Cache for frame skipping
        self._cached_results = None
        
        print(f"\n[Pipeline] Ready. Face={enable_face}, Pose={enable_pose}")
        print(f"[Pipeline] Processing every {self.process_every_n} frames")
        print("=" * 60)
    
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single video frame through the full pipeline.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            FrameResult with all detections, classifications, and annotations
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Frame skip optimization
        should_process = (self.frame_count % self.process_every_n == 0)
        
        if not should_process and self._cached_results is not None:
            # Reuse cached results, just update the annotated frame
            cached = self._cached_results
            annotated = self._draw_annotations(frame, cached.person_detections,
                                                cached.face_detections,
                                                cached.pose_results,
                                                cached.threat_assessments)
            elapsed = (time.time() - start_time) * 1000
            return FrameResult(
                person_detections=cached.person_detections,
                face_detections=cached.face_detections,
                pose_results=cached.pose_results,
                threat_assessments=cached.threat_assessments,
                annotated_frame=annotated,
                inference_time_ms=elapsed,
                person_count=cached.person_count,
                alert_count=cached.alert_count
            )
        
        # ============ FULL PIPELINE ============
        
        # 1. Person Detection (YOLOv8)
        person_detections = self.person_detector.detect(frame)
        
        # 2-4. Face Recognition Pipeline
        face_detections = []
        face_identities = {}  # Maps person index → (name, confidence)
        
        if self.enable_face:
            face_detections = self.face_detector.detect(frame)
            
            if face_detections:
                face_crops = [fd.face_crop for fd in face_detections if fd.face_crop is not None]
                if face_crops:
                    embeddings = self.face_embedder.extract_batch_embeddings(face_crops)
                    
                    for i, (fd, emb) in enumerate(zip(face_detections, embeddings)):
                        if emb is not None:
                            name, conf = self.face_classifier.predict(emb)
                            face_identities[i] = (name, conf)
                        else:
                            face_identities[i] = ("Unknown", 0.0)
        
        # 5-6. Pose & Activity Pipeline
        pose_results = []
        activity_labels = {}  # Maps person index → (activity, confidence)
        
        if self.enable_pose:
            for i, det in enumerate(person_detections):
                pose_result = self.pose_estimator.estimate(frame, det.bbox)
                pose_results.append(pose_result)
                
                if pose_result is not None and self.activity_classifier.is_trained:
                    activity, conf = self.activity_classifier.predict(pose_result.feature_vector)
                    activity_labels[i] = (activity, conf)
                else:
                    activity_labels[i] = ("unknown", 0.0)
        
        # 7. Threat Assessment
        threat_assessments = []
        alert_count = 0
        
        for i, det in enumerate(person_detections):
            # Find matching face (closest face detection to person center)
            face_name, face_conf = "Unknown", 0.0
            if face_identities:
                face_name, face_conf = self._match_face_to_person(
                    det, face_detections, face_identities)
            
            activity, act_conf = activity_labels.get(i, ("unknown", 0.0))
            
            assessment = self.zone_monitor.assess_threat(
                person_bbox=det.bbox,
                face_identity=face_name,
                face_confidence=face_conf,
                activity=activity,
                activity_confidence=act_conf
            )
            threat_assessments.append(assessment)
            
            if assessment.threat_level in ("HIGH", "CRITICAL"):
                alert_count += 1
        
        # Draw annotations
        annotated = self._draw_annotations(frame, person_detections,
                                            face_detections, pose_results,
                                            threat_assessments)
        
        elapsed = (time.time() - start_time) * 1000
        self.total_time += elapsed
        
        result = FrameResult(
            person_detections=person_detections,
            face_detections=face_detections,
            pose_results=pose_results,
            threat_assessments=threat_assessments,
            annotated_frame=annotated,
            inference_time_ms=elapsed,
            person_count=len(person_detections),
            alert_count=alert_count
        )
        
        self._cached_results = result
        return result
    
    def _match_face_to_person(self, person_det: Detection,
                               face_dets: List[FaceDetection],
                               face_ids: dict) -> Tuple[str, float]:
        """Find the face detection closest to the person's bounding box."""
        px, py = person_det.center
        best_dist = float('inf')
        best_id = ("Unknown", 0.0)
        
        for i, fd in enumerate(face_dets):
            fx, fy = fd.center
            dist = np.sqrt((px - fx) ** 2 + (py - fy) ** 2)
            if dist < best_dist and i in face_ids:
                best_dist = dist
                best_id = face_ids[i]
        
        return best_id
    
    def _draw_annotations(self, frame, person_dets, face_dets,
                           pose_results, threat_assessments):
        """Draw all annotations on frame."""
        annotated = frame.copy()
        
        # Draw zone
        annotated = self.zone_monitor.draw_zone(annotated)
        
        # Draw threat assessments (includes person bboxes)
        for assessment in threat_assessments:
            annotated = self.zone_monitor.draw_threat(annotated, assessment)
        
        # Draw pose skeletons
        if self.enable_pose:
            for pr in pose_results:
                if pr is not None:
                    annotated = self.pose_estimator.draw_pose(annotated, pr, (0, 255, 0))
        
        # Draw face landmarks
        if self.enable_face:
            for fd in face_dets:
                if fd.landmarks is not None:
                    for point in fd.landmarks:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(annotated, (x, y), 2, (255, 255, 0), -1)
        
        # HUD overlay
        fps = 1000.0 / max(self.total_time / max(self.frame_count, 1), 1)
        person_count = len(person_dets)
        alert_count = sum(1 for a in threat_assessments if a.threat_level in ("HIGH", "CRITICAL"))
        
        # Background bar
        cv2.rectangle(annotated, (0, 0), (350, 80), (0, 0, 0), -1)
        cv2.putText(annotated, f"Persons: {person_count}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Alerts: {alert_count}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255) if alert_count > 0 else (0, 255, 0), 2)
        cv2.putText(annotated, f"FPS: {fps:.0f}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        return annotated
    
    def get_stats(self) -> dict:
        """Return pipeline performance statistics."""
        avg_ms = self.total_time / max(self.frame_count, 1)
        return {
            "total_frames": self.frame_count,
            "avg_inference_ms": round(avg_ms, 1),
            "avg_fps": round(1000 / max(avg_ms, 1), 1),
            "person_detector_stats": self.person_detector.get_stats()
        }
