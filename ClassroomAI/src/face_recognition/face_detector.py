"""
ClassroomAI - Face Detection & Alignment Module
Locates faces inside detected student bounding boxes or full frames and normalizes them for ArcFace.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("ClassroomAI.FaceDetection")


class FaceDetector:
    """
    Detects faces within student bounding boxes or entire frames.
    Extracts and resizes 112x112 crops normalized for ArcFace feature extraction.
    """

    def __init__(self, min_face_size: Tuple[int, int] = (30, 30)):
        self.min_face_size = min_face_size
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            logger.error("Failed to load OpenCV face Haar Cascade.")
        else:
            logger.info("FaceDetector initialized with Haar Cascade.")

    def detect_face_in_crop(
        self,
        student_crop: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Locates the primary face within a student body crop.
        Falls back to the upper region of the student bounding box if face detector misses
        due to non-frontal pose (common in classroom rear cameras).

        Returns:
            Dict containing 'face_crop' (112x112 RGB), 'box_in_crop' (x, y, w, h), and 'is_fallback' (bool).
        """
        if student_crop is None or student_crop.size == 0:
            return None

        h, w = student_crop.shape[:2]
        gray = cv2.cvtColor(student_crop, cv2.COLOR_BGR2GRAY)
        
        # Upper body region is where the face should appear (top 45% of student bounding box)
        upper_h = max(int(h * 0.45), min(h, 40))
        gray_upper = gray[:upper_h, :]

        faces = self.cascade.detectMultiScale(
            gray_upper,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=self.min_face_size
        )

        if len(faces) > 0:
            # Pick largest detected face in upper region
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces[0]
            face_img = student_crop[fy:fy + fh, fx:fx + fw]
            face_resized = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_LINEAR)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            return {
                "face_crop": face_rgb,
                "box": (fx, fy, fw, fh),
                "is_fallback": False
            }

        # Heuristic fallback: Upper center 30% of student crop
        # Enables feature extraction even when students glance sideways or downwards
        fh_fallback = int(h * 0.28)
        fw_fallback = int(w * 0.55)
        fx_fallback = max(0, (w - fw_fallback) // 2)
        fy_fallback = max(0, int(h * 0.04))

        face_img = student_crop[fy_fallback:fy_fallback + fh_fallback, fx_fallback:fx_fallback + fw_fallback]
        if face_img.size == 0:
            return None

        face_resized = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_LINEAR)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        return {
            "face_crop": face_rgb,
            "box": (fx_fallback, fy_fallback, fw_fallback, fh_fallback),
            "is_fallback": True
        }
