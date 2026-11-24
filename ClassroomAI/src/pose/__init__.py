"""
Pose and Head Orientation Module for ClassroomAI
Integrates 6DRepNet head pose estimation and MediaPipe upper-body posture analysis.
"""

from .head_pose import HeadPoseEstimator, HeadOrientation
from .mediapipe_analyzer import PostureAnalyzer, PostureFeatures
from .feature_extractor import ClassroomFeatureExtractor

__all__ = [
    "HeadPoseEstimator",
    "HeadOrientation",
    "PostureAnalyzer",
    "PostureFeatures",
    "ClassroomFeatureExtractor"
]
