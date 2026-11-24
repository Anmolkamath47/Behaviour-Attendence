"""
Attendance Module for ClassroomAI
Calculates verified physical attendance strictly adhering to the 75% rule.
Maintains clear separation between physical presence and attentiveness.
"""

from .presence_tracker import StudentPresenceTracker, PresenceObservation
from .attendance_engine import AttendanceEngine, AttendanceRecord, AttendanceStatus, AuditLogEntry

__all__ = [
    "StudentPresenceTracker",
    "PresenceObservation",
    "AttendanceEngine",
    "AttendanceRecord",
    "AttendanceStatus",
    "AuditLogEntry"
]
