"""
Tracking Module for ClassroomAI
Implements ByteTrack student tracking, temporary track maintenance, and observation gap recording.
"""

from .student_tracker import StudentTracker, TrackedStudent

__all__ = ["StudentTracker", "TrackedStudent"]
