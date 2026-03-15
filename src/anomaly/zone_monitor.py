"""
Sentinel Vision — Zone Monitor
================================
Combines zone intrusion detection with face identity and activity
classification to produce a unified threat level score.

Key concepts demonstrated:
- Polygon-based region of interest
- Multi-signal threat scoring (weighted fusion)
- Point-in-polygon test (cv2.pointPolygonTest)
"""

import cv2
import numpy as np
import yaml
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ThreatAssessment:
    """Result of threat analysis for a single person."""
    person_bbox: Tuple[int, int, int, int]
    is_in_zone: bool
    face_identity: str
    face_confidence: float
    activity: str
    activity_confidence: float
    threat_score: float  # 0.0 (safe) to 1.0 (max threat)
    threat_level: str    # "LOW", "MEDIUM", "HIGH", "CRITICAL"


class ZoneMonitor:
    """
    Monitors restricted zones and produces threat assessments.
    
    Threat scoring formula:
        score = w1 * unknown_face + w2 * suspicious_activity + w3 * zone_intrusion
    
    All weights are configurable via config.yaml.
    
    Args:
        config_path: Path to config.yaml
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        zone_config = config['zone']
        
        self.max_points = zone_config['max_points']
        self.alert_cooldown = zone_config['alert_cooldown_seconds']
        self.weights = zone_config['threat_weights']
        self.threat_threshold = zone_config['threat_threshold']
        
        self.zone_points: List[List[int]] = []
        self.threat_history: List[float] = []
    
    @property
    def zone_active(self) -> bool:
        """Whether a complete zone polygon is defined."""
        return len(self.zone_points) == self.max_points
    
    def add_point(self, x: int, y: int) -> bool:
        """Add a point to the zone polygon. Returns True if zone became active."""
        if len(self.zone_points) < self.max_points:
            self.zone_points.append([x, y])
            return self.zone_active
        return False
    
    def clear_zone(self):
        """Clear all zone points."""
        self.zone_points.clear()
    
    def is_point_in_zone(self, x: int, y: int) -> bool:
        """Test if a point is inside the restricted zone polygon."""
        if not self.zone_active:
            return False
        
        pts = np.array(self.zone_points, np.int32)
        result = cv2.pointPolygonTest(pts, (x, y), False)
        return result >= 0
    
    def assess_threat(self, person_bbox: Tuple[int, int, int, int],
                       face_identity: str = "Unknown",
                       face_confidence: float = 0.0,
                       activity: str = "normal_walking",
                       activity_confidence: float = 0.0) -> ThreatAssessment:
        """
        Assess threat level for a detected person.
        
        Combines three signals:
        1. Unknown face → higher threat
        2. Suspicious activity → higher threat
        3. Inside restricted zone → higher threat
        
        Args:
            person_bbox: (x1, y1, x2, y2)
            face_identity: Name or "Unknown"
            face_confidence: Face classification confidence
            activity: Activity label
            activity_confidence: Activity classification confidence
            
        Returns:
            ThreatAssessment with composite threat score
        """
        x1, y1, x2, y2 = person_bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Signal 1: Unknown face
        unknown_score = 1.0 if face_identity == "Unknown" else 0.0
        
        # Signal 2: Suspicious activity
        suspicious_activities = {'suspicious', 'running', 'falling', 'fighting'}
        if activity in suspicious_activities:
            activity_score = activity_confidence
        elif activity == 'loitering':
            activity_score = activity_confidence * 0.5
        else:
            activity_score = 0.0
        
        # Signal 3: Zone intrusion
        in_zone = self.is_point_in_zone(cx, cy)
        zone_score = 1.0 if in_zone else 0.0
        
        # Weighted combination
        threat_score = (
            self.weights['unknown_face'] * unknown_score +
            self.weights['suspicious_activity'] * activity_score +
            self.weights['zone_intrusion'] * zone_score
        )
        threat_score = min(1.0, max(0.0, threat_score))
        
        # Determine threat level
        if threat_score >= 0.8:
            threat_level = "CRITICAL"
        elif threat_score >= 0.6:
            threat_level = "HIGH"
        elif threat_score >= 0.3:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"
        
        self.threat_history.append(threat_score)
        
        return ThreatAssessment(
            person_bbox=person_bbox,
            is_in_zone=in_zone,
            face_identity=face_identity,
            face_confidence=face_confidence,
            activity=activity,
            activity_confidence=activity_confidence,
            threat_score=threat_score,
            threat_level=threat_level
        )
    
    def draw_zone(self, frame: np.ndarray) -> np.ndarray:
        """Draw the restricted zone overlay on a frame."""
        annotated = frame.copy()
        
        for pt in self.zone_points:
            cv2.circle(annotated, (pt[0], pt[1]), 8, (0, 0, 255), -1)
        
        if len(self.zone_points) > 1:
            pts_array = np.array(self.zone_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts_array], self.zone_active, (0, 255, 255), 2)
        
        if self.zone_active:
            overlay = annotated.copy()
            pts_array = np.array(self.zone_points, np.int32)
            cv2.fillPoly(overlay, [pts_array], (0, 255, 255))
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
        
        return annotated
    
    def draw_threat(self, frame: np.ndarray, assessment: ThreatAssessment) -> np.ndarray:
        """Draw threat assessment annotations on frame."""
        annotated = frame.copy()
        x1, y1, x2, y2 = assessment.person_bbox
        
        # Color based on threat level
        colors = {
            "LOW": (0, 255, 0),
            "MEDIUM": (0, 165, 255),
            "HIGH": (0, 0, 255),
            "CRITICAL": (0, 0, 200)
        }
        color = colors.get(assessment.threat_level, (255, 255, 255))
        
        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Labels
        label1 = f"{assessment.face_identity} ({assessment.face_confidence:.0%})"
        label2 = f"{assessment.activity} | {assessment.threat_level}"
        
        cv2.putText(annotated, label1, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(annotated, label2, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Alert overlay for critical threats
        if assessment.threat_level == "CRITICAL":
            h, w = annotated.shape[:2]
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
            cv2.putText(annotated, "! CRITICAL THREAT !", (50, 50),
                       cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 3)
        
        return annotated
