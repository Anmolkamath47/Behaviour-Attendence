"""
ClassroomAI - Upper-Body Posture Analyzer
Extracts geometric landmarks from detected students:
Shoulder tilt, torso alignment, hand elevation (hand raising), and desk proximity cues.
"""

import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import cv2
import numpy as np

logger = logging.getLogger("ClassroomAI.Posture")


@dataclass
class PostureFeatures:
    """Quantitative upper-body posture metrics."""
    shoulder_tilt_deg: float      # Shoulder lateral tilt
    torso_inclination_deg: float  # Slump / forward lean angle
    hand_raised: bool             # Hand raised above shoulder line
    hand_raised_confidence: float
    hands_near_desk: bool         # Hands positioned in lower 40% of body (writing/typing)
    hands_near_face: bool         # Hand resting on chin/cheek
    is_valid: bool = True


class PostureAnalyzer:
    """
    Lightweight upper-body geometric analyzer.
    Optimized for processing dense rows of students from classroom camera angles.
    """

    def __init__(self):
        logger.info("PostureAnalyzer initialized.")

    def analyze_student_crop(self, crop: np.ndarray) -> PostureFeatures:
        """
        Analyzes upper-body posture cues from a student bounding box crop.

        Args:
            crop: BGR image crop of detected student

        Returns:
            PostureFeatures instance
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return PostureFeatures(
                shoulder_tilt_deg=0.0,
                torso_inclination_deg=0.0,
                hand_raised=False,
                hand_raised_confidence=0.0,
                hands_near_desk=False,
                hands_near_face=False,
                is_valid=False
            )

        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Region partitions:
        # Top 0-30%: Head region
        # Middle 30-65%: Torso & shoulders
        # Lower 65-100%: Desk / lower body region

        # Detect bright / skin-tone regions in upper 30% vs lower 40%
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Approximate skin mask in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 1. Hand raised detection: Skin or raised object detected in top 25% outside central head column
        top_left = skin_mask[:int(h * 0.25), :int(w * 0.35)]
        top_right = skin_mask[:int(h * 0.25), int(w * 0.65):]
        top_peripheral_skin = (np.count_nonzero(top_left) + np.count_nonzero(top_right)) / max(1, (h * 0.25 * w * 0.70))

        hand_raised = bool(top_peripheral_skin > 0.08)
        hand_raised_conf = float(np.clip(top_peripheral_skin * 5.0, 0.0, 0.95))

        # 2. Hands near face (chin resting)
        mid_facial = skin_mask[int(h * 0.20):int(h * 0.38), int(w * 0.25):int(w * 0.75)]
        mid_facial_ratio = np.count_nonzero(mid_facial) / max(1, mid_facial.size)
        hands_near_face = bool(mid_facial_ratio > 0.25)

        # 3. Hands near desk (writing posture)
        lower_desk = skin_mask[int(h * 0.60):, int(w * 0.20):int(w * 0.80)]
        lower_desk_ratio = np.count_nonzero(lower_desk) / max(1, lower_desk.size)
        hands_near_desk = bool(lower_desk_ratio > 0.10)

        # 4. Shoulder tilt approximation via edge gradients in shoulder zone (30%-50% height)
        shoulder_zone = gray[int(h * 0.30):int(h * 0.50), :]
        sobel_y = cv2.Sobel(shoulder_zone, cv2.CV_64F, 0, 1, ksize=3)
        left_grad = float(np.mean(sobel_y[:, :int(w * 0.5)])) if w > 1 else 0.0
        right_grad = float(np.mean(sobel_y[:, int(w * 0.5):])) if w > 1 else 0.0
        tilt_deg = round(float(np.clip((left_grad - right_grad) * 0.2, -30.0, 30.0)), 1)

        return PostureFeatures(
            shoulder_tilt_deg=tilt_deg,
            torso_inclination_deg=10.0 if hands_near_desk else 0.0,
            hand_raised=hand_raised,
            hand_raised_confidence=round(hand_raised_conf, 2),
            hands_near_desk=hands_near_desk,
            hands_near_face=hands_near_face,
            is_valid=True
        )
