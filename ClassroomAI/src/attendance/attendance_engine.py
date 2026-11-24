"""
ClassroomAI - Attendance Automation Engine (Strict 75% Rule)
Implements:
Presence Percentage = (Verified Presence Minutes / Total Lecture Minutes) * 100

Decision Rules:
- >= 75.0% Verified Presence: PRESENT
- < 75.0%: BELOW_REQUIRED_ATTENDANCE
- High observation uncertainty / camera interruption: NEEDS_MANUAL_REVIEW

IMPORTANT: Physical presence and attentiveness are completely decoupled.
A student writing notes or looking down is NEVER marked absent on behavioural grounds.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from .presence_tracker import StudentPresenceTracker

logger = logging.getLogger("ClassroomAI.AttendanceEngine")

ATTENDANCE_THRESHOLD_PERCENT = 75.0


class AttendanceStatus(str, Enum):
    PRESENT = "PRESENT"
    BELOW_REQUIRED = "BELOW_REQUIRED_ATTENDANCE"
    NEEDS_MANUAL_REVIEW = "NEEDS_MANUAL_REVIEW"


@dataclass
class AuditLogEntry:
    """Immutable audit trail entry for attendance decisions or manual corrections."""
    timestamp: str
    student_id: str
    previous_status: str
    new_status: str
    modified_by: str
    reason: str


@dataclass
class AttendanceRecord:
    """Official attendance report for an enrolled student for a lecture session."""
    session_id: str
    student_id: str
    roll_number: str
    student_name: str
    total_lecture_minutes: float
    verified_presence_minutes: float
    uncertain_minutes: float
    presence_percentage: float
    status: AttendanceStatus
    arrival_time: Optional[str]
    departure_time: Optional[str]
    manual_review_required: bool
    manual_override: bool = False
    audit_notes: Optional[str] = None


class AttendanceEngine:
    """
    Computes verified attendance for classroom lectures.
    Preserves audit history and enforces the 75% presence threshold.
    """

    def __init__(self, session_id: str, lecture_duration_minutes: float = 60.0):
        self.session_id = session_id
        self.total_lecture_minutes = max(1.0, lecture_duration_minutes)
        self.trackers: Dict[str, StudentPresenceTracker] = {}
        self.student_names: Dict[str, str] = {}
        self.audit_log: List[AuditLogEntry] = []
        self.finalized_records: Dict[str, AttendanceRecord] = {}

    def register_student(self, student_id: str, roll_number: str, name: str):
        """Enroll student tracker in session."""
        if student_id not in self.trackers:
            self.trackers[student_id] = StudentPresenceTracker(student_id, roll_number)
            self.student_names[student_id] = name

    def record_presence(
        self,
        student_id: str,
        timestamp: float,
        is_verified_identity: bool,
        confidence: float
    ):
        """Record a physical presence observation."""
        if student_id in self.trackers:
            self.trackers[student_id].record_observation(
                timestamp=timestamp,
                is_verified_identity=is_verified_identity,
                confidence=confidence
            )

    def calculate_attendance(self, student_id: str) -> AttendanceRecord:
        """
        Calculates verified attendance adhering strictly to the 75% rule.

        Formula:
        Presence Percentage = (Verified Presence Minutes / Total Lecture Minutes) * 100
        """
        tracker = self.trackers.get(student_id)
        name = self.student_names.get(student_id, "Unknown")
        roll_no = tracker.roll_number if tracker else "N/A"

        if not tracker:
            return AttendanceRecord(
                session_id=self.session_id,
                student_id=student_id,
                roll_number=roll_no,
                student_name=name,
                total_lecture_minutes=self.total_lecture_minutes,
                verified_presence_minutes=0.0,
                uncertain_minutes=0.0,
                presence_percentage=0.0,
                status=AttendanceStatus.BELOW_REQUIRED,
                arrival_time=None,
                departure_time=None,
                manual_review_required=True,
                audit_notes="No presence recorded."
            )

        verified_min = round(tracker.verified_minutes, 2)
        uncertain_min = round(tracker.uncertain_minutes, 2)
        presence_pct = round((verified_min / self.total_lecture_minutes) * 100.0, 1)

        # Decision logic
        if uncertain_min > (self.total_lecture_minutes * 0.25):
            # If camera observations were uncertain for > 25% of the class
            status = AttendanceStatus.NEEDS_MANUAL_REVIEW
            review_required = True
            notes = "High uncertainty in camera recognition; requires teacher verification."
        elif presence_pct >= ATTENDANCE_THRESHOLD_PERCENT:
            status = AttendanceStatus.PRESENT
            review_required = False
            notes = f"Verified presence {presence_pct}% meets >= {ATTENDANCE_THRESHOLD_PERCENT}% threshold."
        else:
            status = AttendanceStatus.BELOW_REQUIRED
            review_required = False
            notes = f"Verified presence {presence_pct}% is below required {ATTENDANCE_THRESHOLD_PERCENT}%."

        record = AttendanceRecord(
            session_id=self.session_id,
            student_id=student_id,
            roll_number=roll_no,
            student_name=name,
            total_lecture_minutes=self.total_lecture_minutes,
            verified_presence_minutes=verified_min,
            uncertain_minutes=uncertain_min,
            presence_percentage=presence_pct,
            status=status,
            arrival_time=tracker.arrival_time_str,
            departure_time=tracker.departure_time_str,
            manual_review_required=review_required,
            audit_notes=notes
        )

        self.finalized_records[student_id] = record
        return record

    def apply_manual_override(
        self,
        student_id: str,
        new_status: AttendanceStatus,
        teacher_id: str,
        reason: str
    ) -> bool:
        """
        Allows a teacher to manually override an attendance status with full audit logging.
        """
        if student_id not in self.finalized_records:
            self.calculate_attendance(student_id)

        record = self.finalized_records[student_id]
        old_status = record.status.value

        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            student_id=student_id,
            previous_status=old_status,
            new_status=new_status.value,
            modified_by=teacher_id,
            reason=reason
        )
        self.audit_log.append(entry)

        record.status = new_status
        record.manual_override = True
        record.manual_review_required = False
        record.audit_notes = f"Overridden by Teacher {teacher_id}: {reason}"

        logger.info(
            f"[Audit] Attendance override for Student {record.roll_number}: "
            f"{old_status} -> {new_status.value} by {teacher_id}. Reason: {reason}"
        )
        return True

    def generate_session_summary(self) -> Dict[str, Any]:
        """Generate classroom-wide attendance statistics."""
        for s_id in self.trackers.keys():
            if s_id not in self.finalized_records:
                self.calculate_attendance(s_id)

        records = list(self.finalized_records.values())
        total = len(records)
        present = sum(1 for r in records if r.status == AttendanceStatus.PRESENT)
        below = sum(1 for r in records if r.status == AttendanceStatus.BELOW_REQUIRED)
        review = sum(1 for r in records if r.status == AttendanceStatus.NEEDS_MANUAL_REVIEW)

        class_rate = round((present / max(1, total)) * 100.0, 1)

        return {
            "session_id": self.session_id,
            "total_students": total,
            "present_count": present,
            "below_required_count": below,
            "needs_review_count": review,
            "class_attendance_rate": class_rate,
            "total_audit_events": len(self.audit_log)
        }
