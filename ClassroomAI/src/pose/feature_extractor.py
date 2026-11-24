"""
ClassroomAI - Multi-Modal Classroom Feature Extractor
Combines YOLO bounding boxes, 6DRepNet head pose, and MediaPipe posture into a unified
12-dimensional feature vector per student for the TSM and XGBoost classifiers.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np

from .head_pose import HeadPoseEstimator, HeadOrientation
from .mediapipe_analyzer import PostureAnalyzer, PostureFeatures

logger = logging.getLogger("ClassroomAI.FeatureExtractor")

FEATURE_NAMES = [
    "head_pitch",             # -45 to 45 deg (down/up)
    "head_yaw",               # -60 to 60 deg (turn left/right)
    "head_roll",              # -30 to 30 deg (lateral tilt)
    "shoulder_tilt_deg",      # -30 to 30 deg
    "torso_inclination_deg",  # 0 to 45 deg
    "is_looking_down",        # 1.0 or 0.0
    "is_facing_front",        # 1.0 or 0.0
    "is_looking_away",        # 1.0 or 0.0
    "hand_raised_conf",       # 0.0 to 1.0
    "hands_near_desk",        # 1.0 or 0.0
    "hands_near_face",        # 1.0 or 0.0
    "phone_detected_nearby"   # 1.0 or 0.0 (from YOLO context)
]


class ClassroomFeatureExtractor:
    """
    Extracts structured 12-D multi-modal feature vectors for each tracked student.
    """

    def __init__(self):
        self.head_pose_estimator = HeadPoseEstimator()
        self.posture_analyzer = PostureAnalyzer()

    def extract_features(
        self,
        student_crop: np.ndarray,
        phone_nearby: bool = False
    ) -> Dict[str, Any]:
        """
        Extract complete posture, head orientation, and context feature dictionary and vector.

        Returns:
            Dict containing 'vector' (12-D np.ndarray), 'head_orientation', and 'posture'.
        """
        head_orientation = self.head_pose_estimator.estimate_from_crop(student_crop)
        posture = self.posture_analyzer.analyze_student_crop(student_crop)

        is_looking_down = 1.0 if head_orientation.direction == "LOOKING_DOWN" else 0.0
        is_facing_front = 1.0 if head_orientation.direction == "FACING_FORWARD" else 0.0
        is_looking_away = 1.0 if head_orientation.direction in ("LOOKING_LEFT", "LOOKING_RIGHT") else 0.0

        vector = np.array([
            head_orientation.pitch,
            head_orientation.yaw,
            head_orientation.roll,
            posture.shoulder_tilt_deg,
            posture.torso_inclination_deg,
            is_looking_down,
            is_facing_front,
            is_looking_away,
            posture.hand_raised_confidence,
            1.0 if posture.hands_near_desk else 0.0,
            1.0 if posture.hands_near_face else 0.0,
            1.0 if phone_nearby else 0.0
        ], dtype=np.float32)

        return {
            "vector": vector,
            "feature_names": FEATURE_NAMES,
            "head_orientation": head_orientation,
            "posture": posture
        }
