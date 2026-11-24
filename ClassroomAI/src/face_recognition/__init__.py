"""
Face Recognition Module for ClassroomAI
Implements face detection, ArcFace 512-D embeddings, and identity matching with informed consent verification.
"""

from .face_detector import FaceDetector
from .arcface import ArcFaceEmbedder
from .matcher import FaceMatcher, StudentProfile, MatchResult

__all__ = ["FaceDetector", "ArcFaceEmbedder", "FaceMatcher", "StudentProfile", "MatchResult"]
