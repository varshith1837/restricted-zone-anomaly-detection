"""
Sentinel Vision — Alert Manager
==================================
Centralized alert handling with cooldowns, screenshot capture,
and event logging to CSV.
"""

import os
import cv2
import csv
import datetime
import yaml
from typing import Optional
from src.anomaly.zone_monitor import ThreatAssessment


class AlertManager:
    """
    Manages alerts, screenshots, and event logging.
    
    Features:
    - Cooldown-based alert throttling
    - Auto screenshot capture for HIGH/CRITICAL threats
    - CSV event logging with metadata
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.intruders_dir = config['paths']['intruders_dir']
        self.cooldown_seconds = config['zone']['alert_cooldown_seconds']
        self.threat_threshold = config['zone']['threat_threshold']
        
        os.makedirs(self.intruders_dir, exist_ok=True)
        
        self.last_alert_time = datetime.datetime.now() - datetime.timedelta(seconds=self.cooldown_seconds + 1)
        self.total_alerts = 0
        self.log_file = os.path.join(self.intruders_dir, "alert_log.csv")
        
        # Initialize log file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'identity', 'activity', 'threat_score',
                                 'threat_level', 'in_zone', 'screenshot_path'])
    
    def process_alert(self, assessment: ThreatAssessment,
                       frame: Optional['numpy.ndarray'] = None) -> bool:
        """
        Process a threat assessment and handle alerting.
        
        Args:
            assessment: ThreatAssessment from ZoneMonitor
            frame: Current video frame for screenshot capture
            
        Returns:
            True if alert was triggered, False if suppressed by cooldown
        """
        if assessment.threat_score < self.threat_threshold:
            return False
        
        now = datetime.datetime.now()
        elapsed = (now - self.last_alert_time).total_seconds()
        
        if elapsed < self.cooldown_seconds:
            return False
        
        self.last_alert_time = now
        self.total_alerts += 1
        
        # Save screenshot
        screenshot_path = ""
        if frame is not None:
            timestamp_str = now.strftime('%Y%m%d_%H%M%S')
            filename = f"Alert_{timestamp_str}_{assessment.threat_level}.jpg"
            screenshot_path = os.path.join(self.intruders_dir, filename)
            cv2.imwrite(screenshot_path, frame)
        
        # Log event
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime('%Y-%m-%d %H:%M:%S'),
                assessment.face_identity,
                assessment.activity,
                f"{assessment.threat_score:.3f}",
                assessment.threat_level,
                assessment.is_in_zone,
                screenshot_path
            ])
        
        print(f"[ALERT] {assessment.threat_level} — {assessment.face_identity} "
              f"({assessment.activity}) — Score: {assessment.threat_score:.2f}")
        
        return True
    
    def get_stats(self) -> dict:
        return {
            "total_alerts": self.total_alerts,
            "cooldown": self.cooldown_seconds,
            "log_file": self.log_file
        }
