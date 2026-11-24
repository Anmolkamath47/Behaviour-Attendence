"""
ClassroomAI - Database Package Init
"""

from .connection import engine, SessionLocal, Base, get_db
from .models import (
    User, Student, Classroom, ClassroomStudent, SeatingAssignment,
    ConsentRecord, LectureSession, AttendanceRecord, PresenceObservation,
    BehaviourObservation, ModelVersion, AuditLog
)

__all__ = [
    "engine", "SessionLocal", "Base", "get_db",
    "User", "Student", "Classroom", "ClassroomStudent", "SeatingAssignment",
    "ConsentRecord", "LectureSession", "AttendanceRecord", "PresenceObservation",
    "BehaviourObservation", "ModelVersion", "AuditLog"
]
