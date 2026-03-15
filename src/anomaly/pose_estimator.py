"""
Sentinel Vision — MediaPipe Pose Estimator
=============================================
Extracts body pose landmarks for activity classification.

Key ML concepts demonstrated:
- Pose estimation with BlazePose (MediaPipe)
- Feature engineering from raw landmarks
- Normalization for translation/scale invariance
- Temporal feature extraction (velocity, acceleration)
"""

import cv2
import numpy as np
import yaml
import math
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class PoseResult:
    """Result of pose estimation for a single person."""
    landmarks: np.ndarray           # (33, 3) — x, y, visibility per landmark
    normalized_landmarks: np.ndarray  # Hip-centered, scale-normalized
    feature_vector: np.ndarray      # Engineered features for classification
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # Bounding box of the person
    confidence: float = 0.0


class PoseEstimator:
    """
    MediaPipe-based body pose estimator.
    
    Extracts 33 body landmarks per person and engineers classification-ready
    features:
    - Joint angles (shoulder, elbow, knee, hip)
    - Inter-keypoint distances (arm span, leg spread, etc.)
    - Aspect ratio of bounding box
    - Center of mass position
    
    These features are fed to the ActivityClassifier.
    
    Args:
        config_path: Path to config.yaml
    """
    
    # MediaPipe landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        pose_config = config['pose']
        
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "mediapipe not installed. Run: pip install mediapipe"
            )
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=pose_config['model_complexity'],
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=pose_config['min_detection_confidence'],
            min_tracking_confidence=pose_config['min_tracking_confidence']
        )
        
        # History buffer for temporal features
        self.landmark_history: List[np.ndarray] = []
        self.max_history = config['activity'].get('feature_window', 15)
        
        print(f"[PoseEstimator] MediaPipe Pose loaded (complexity={pose_config['model_complexity']})")
    
    @staticmethod
    def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculate angle at point b, formed by points a-b-c.
        
        Args:
            a, b, c: 2D or 3D coordinates
            
        Returns:
            Angle in degrees (0-180)
        """
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        return float(np.degrees(np.arccos(cosine_angle)))
    
    @staticmethod
    def _calc_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Euclidean distance between two points."""
        return float(np.linalg.norm(p1[:2] - p2[:2]))
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to be translation and scale invariant.
        
        Centers on mid-hip and scales by torso length.
        
        Args:
            landmarks: (33, 3) raw landmarks
            
        Returns:
            (33, 3) normalized landmarks
        """
        # Center of hips
        hip_center = (landmarks[self.LEFT_HIP] + landmarks[self.RIGHT_HIP]) / 2.0
        
        # Translate to hip center
        normalized = landmarks.copy()
        normalized[:, :2] -= hip_center[:2]
        
        # Scale by torso length (hip center to shoulder midpoint)
        shoulder_center = (landmarks[self.LEFT_SHOULDER] + landmarks[self.RIGHT_SHOULDER]) / 2.0
        torso_length = self._calc_distance(hip_center, shoulder_center)
        
        if torso_length > 1e-6:
            normalized[:, :2] /= torso_length
        
        return normalized
    
    def _extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Engineer classification features from raw landmarks.
        
        Features extracted:
        1. Joint angles (6 angles)
        2. Key distances (5 distances)
        3. Bounding box aspect ratio (1)
        4. Center of mass vertical position (1)
        5. Arm raise indicators (2)
        6. Temporal features if history available (4)
        
        Total: ~19 features
        
        Args:
            landmarks: (33, 3) normalized landmarks
            
        Returns:
            Feature vector (numpy array)
        """
        features = []
        
        # ---- Joint Angles (6) ----
        # Left elbow angle
        features.append(self._calc_angle(
            landmarks[self.LEFT_SHOULDER],
            landmarks[self.LEFT_ELBOW],
            landmarks[self.LEFT_WRIST]
        ))
        
        # Right elbow angle
        features.append(self._calc_angle(
            landmarks[self.RIGHT_SHOULDER],
            landmarks[self.RIGHT_ELBOW],
            landmarks[self.RIGHT_WRIST]
        ))
        
        # Left shoulder angle
        features.append(self._calc_angle(
            landmarks[self.LEFT_HIP],
            landmarks[self.LEFT_SHOULDER],
            landmarks[self.LEFT_ELBOW]
        ))
        
        # Right shoulder angle
        features.append(self._calc_angle(
            landmarks[self.RIGHT_HIP],
            landmarks[self.RIGHT_SHOULDER],
            landmarks[self.RIGHT_ELBOW]
        ))
        
        # Left knee angle
        features.append(self._calc_angle(
            landmarks[self.LEFT_HIP],
            landmarks[self.LEFT_KNEE],
            landmarks[self.LEFT_ANKLE]
        ))
        
        # Right knee angle
        features.append(self._calc_angle(
            landmarks[self.RIGHT_HIP],
            landmarks[self.RIGHT_KNEE],
            landmarks[self.RIGHT_ANKLE]
        ))
        
        # ---- Key Distances (5) ----
        # Shoulder width
        features.append(self._calc_distance(
            landmarks[self.LEFT_SHOULDER],
            landmarks[self.RIGHT_SHOULDER]
        ))
        
        # Hip width
        features.append(self._calc_distance(
            landmarks[self.LEFT_HIP],
            landmarks[self.RIGHT_HIP]
        ))
        
        # Left arm length
        features.append(self._calc_distance(
            landmarks[self.LEFT_SHOULDER],
            landmarks[self.LEFT_WRIST]
        ))
        
        # Right arm length
        features.append(self._calc_distance(
            landmarks[self.RIGHT_SHOULDER],
            landmarks[self.RIGHT_WRIST]
        ))
        
        # Foot spread
        features.append(self._calc_distance(
            landmarks[self.LEFT_ANKLE],
            landmarks[self.RIGHT_ANKLE]
        ))
        
        # ---- Bounding Box Aspect Ratio (1) ----
        all_x = landmarks[:, 0]
        all_y = landmarks[:, 1]
        bbox_w = np.max(all_x) - np.min(all_x) + 1e-8
        bbox_h = np.max(all_y) - np.min(all_y) + 1e-8
        features.append(bbox_h / bbox_w)  # Aspect ratio
        
        # ---- Center of Mass Vertical (1) ----
        com_y = np.mean(landmarks[:, 1])
        features.append(com_y)
        
        # ---- Arm Raise Indicators (2) ----
        # Left hand above shoulder
        features.append(float(landmarks[self.LEFT_WRIST][1] < landmarks[self.LEFT_SHOULDER][1]))
        # Right hand above shoulder
        features.append(float(landmarks[self.RIGHT_WRIST][1] < landmarks[self.RIGHT_SHOULDER][1]))
        
        # ---- Temporal Features (4) ----
        if len(self.landmark_history) >= 2:
            prev = self.landmark_history[-1]
            
            # Overall body velocity (mean displacement)
            displacement = np.mean(np.abs(landmarks[:, :2] - prev[:, :2]))
            features.append(displacement)
            
            # Hip velocity (proxy for walking/running speed)
            hip_curr = (landmarks[self.LEFT_HIP] + landmarks[self.RIGHT_HIP]) / 2
            hip_prev = (prev[self.LEFT_HIP] + prev[self.RIGHT_HIP]) / 2
            features.append(self._calc_distance(hip_curr, hip_prev))
            
            # Upper body movement (arms/head)
            upper_curr = np.mean(landmarks[[self.NOSE, self.LEFT_WRIST, self.RIGHT_WRIST], :2], axis=0)
            upper_prev = np.mean(prev[[self.NOSE, self.LEFT_WRIST, self.RIGHT_WRIST], :2], axis=0)
            features.append(float(np.linalg.norm(upper_curr - upper_prev)))
            
            # Pose stability (variance of recent movement)
            if len(self.landmark_history) >= 5:
                recent = np.array(self.landmark_history[-5:])
                stability = np.mean(np.std(recent[:, :, :2], axis=0))
                features.append(stability)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def estimate(self, frame: np.ndarray, person_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[PoseResult]:
        """
        Estimate pose for a person in the frame.
        
        Args:
            frame: BGR image
            person_bbox: (x1, y1, x2, y2) bounding box to crop from.
                         If None, uses full frame.
            
        Returns:
            PoseResult with landmarks and features, or None if no pose detected
        """
        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame
            person_bbox = (0, 0, frame.shape[1], frame.shape[0])
        
        if roi.size == 0:
            return None
        
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks is None:
            return None
        
        # Extract landmarks as numpy array
        h, w = roi.shape[:2]
        landmarks = np.zeros((33, 3), dtype=np.float32)
        
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[i] = [lm.x * w, lm.y * h, lm.visibility]
        
        # Normalize
        normalized = self._normalize_landmarks(landmarks)
        
        # Extract engineered features
        feature_vector = self._extract_features(normalized)
        
        # Update history
        self.landmark_history.append(normalized.copy())
        if len(self.landmark_history) > self.max_history:
            self.landmark_history.pop(0)
        
        # Compute overall confidence
        confidence = float(np.mean(landmarks[:, 2]))
        
        return PoseResult(
            landmarks=landmarks,
            normalized_landmarks=normalized,
            feature_vector=feature_vector,
            bbox=person_bbox,
            confidence=confidence
        )
    
    def draw_pose(self, frame: np.ndarray, pose_result: PoseResult,
                   color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw pose skeleton on frame.
        
        Args:
            frame: BGR image
            pose_result: PoseResult from estimate()
            color: Skeleton color
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        landmarks = pose_result.landmarks
        
        # Define connections for skeleton
        connections = [
            (self.LEFT_SHOULDER, self.RIGHT_SHOULDER),
            (self.LEFT_SHOULDER, self.LEFT_ELBOW),
            (self.LEFT_ELBOW, self.LEFT_WRIST),
            (self.RIGHT_SHOULDER, self.RIGHT_ELBOW),
            (self.RIGHT_ELBOW, self.RIGHT_WRIST),
            (self.LEFT_SHOULDER, self.LEFT_HIP),
            (self.RIGHT_SHOULDER, self.RIGHT_HIP),
            (self.LEFT_HIP, self.RIGHT_HIP),
            (self.LEFT_HIP, self.LEFT_KNEE),
            (self.LEFT_KNEE, self.LEFT_ANKLE),
            (self.RIGHT_HIP, self.RIGHT_KNEE),
            (self.RIGHT_KNEE, self.RIGHT_ANKLE),
        ]
        
        # Offset by bounding box
        x_off, y_off = pose_result.bbox[0], pose_result.bbox[1]
        
        # Draw connections
        for start, end in connections:
            if landmarks[start][2] > 0.3 and landmarks[end][2] > 0.3:
                pt1 = (int(landmarks[start][0]) + x_off, int(landmarks[start][1]) + y_off)
                pt2 = (int(landmarks[end][0]) + x_off, int(landmarks[end][1]) + y_off)
                cv2.line(annotated, pt1, pt2, color, 2)
        
        # Draw key joints
        key_joints = [self.NOSE, self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
                      self.LEFT_ELBOW, self.RIGHT_ELBOW, self.LEFT_WRIST,
                      self.RIGHT_WRIST, self.LEFT_HIP, self.RIGHT_HIP,
                      self.LEFT_KNEE, self.RIGHT_KNEE, self.LEFT_ANKLE,
                      self.RIGHT_ANKLE]
        
        for idx in key_joints:
            if landmarks[idx][2] > 0.3:
                pt = (int(landmarks[idx][0]) + x_off, int(landmarks[idx][1]) + y_off)
                cv2.circle(annotated, pt, 4, (0, 0, 255), -1)
        
        return annotated
    
    def reset_history(self):
        """Clear landmark history (call when tracking a new person)."""
        self.landmark_history.clear()
    
    def get_feature_names(self) -> List[str]:
        """Return names of engineered features (for analysis)."""
        return [
            'left_elbow_angle', 'right_elbow_angle',
            'left_shoulder_angle', 'right_shoulder_angle',
            'left_knee_angle', 'right_knee_angle',
            'shoulder_width', 'hip_width',
            'left_arm_length', 'right_arm_length',
            'foot_spread',
            'bbox_aspect_ratio', 'com_vertical',
            'left_arm_raised', 'right_arm_raised',
            'body_velocity', 'hip_velocity',
            'upper_body_movement', 'pose_stability'
        ]


# ==========================================
# Standalone test
# ==========================================
if __name__ == "__main__":
    import time
    
    print("=" * 50)
    print("PoseEstimator — Standalone Test")
    print("=" * 50)
    
    estimator = PoseEstimator()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit()
    
    print("Running pose estimation... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        result = estimator.estimate(frame)
        fps = 1.0 / max(time.time() - start, 1e-6)
        
        if result is not None:
            frame = estimator.draw_pose(frame, result)
            cv2.putText(frame, f"Features: {len(result.feature_vector)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Confidence: {result.confidence:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Pose Estimation", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
