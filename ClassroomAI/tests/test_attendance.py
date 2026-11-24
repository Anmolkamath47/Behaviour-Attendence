"""
Unit tests for Attendance Module:
- Strict 75% rule calculation: (verified_minutes / lecture_minutes) * 100
- Prevention of double counting within identical time bins
- Exclusion of uncertain identities from verified attendance
- Teacher manual override and audit trail preservation
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attendance.attendance_engine import AttendanceEngine, AttendanceStatus
from src.attendance.presence_tracker import StudentPresenceTracker


class TestAttendanceEngine(unittest.TestCase):

    def setUp(self):
        self.engine = AttendanceEngine(session_id="LEC-101", lecture_duration_minutes=60.0)

    def test_75_percent_threshold_present(self):
        """
        Verify exact specification example:
        Lecture = 60 min. Student reliably observed for 48 min (2880 sec).
        Presence = (48/60) * 100 = 80.0% -> PRESENT
        """
        self.engine.register_student("STU-1", "CS-01", "Student One")

        # Simulate 2880 distinct seconds (48 minutes) of verified presence
        for sec in range(2880):
            self.engine.record_presence(
                student_id="STU-1",
                timestamp=float(sec),
                is_verified_identity=True,
                confidence=0.92
            )

        record = self.engine.calculate_attendance("STU-1")
        self.assertEqual(record.status, AttendanceStatus.PRESENT)
        self.assertEqual(record.verified_presence_minutes, 48.0)
        self.assertEqual(record.presence_percentage, 80.0)
        self.assertFalse(record.manual_review_required)

    def test_75_percent_threshold_below_required(self):
        """
        Verify student below 75% threshold:
        Lecture = 60 min. Student observed for 30 min (1800 sec).
        Presence = (30/60) * 100 = 50.0% -> BELOW_REQUIRED_ATTENDANCE
        """
        self.engine.register_student("STU-2", "CS-02", "Student Two")

        for sec in range(1800):
            self.engine.record_presence(
                student_id="STU-2",
                timestamp=float(sec),
                is_verified_identity=True,
                confidence=0.88
            )

        record = self.engine.calculate_attendance("STU-2")
        self.assertEqual(record.status, AttendanceStatus.BELOW_REQUIRED)
        self.assertEqual(record.presence_percentage, 50.0)

    def test_prevention_of_double_counting(self):
        """
        Verify multiple observations in the same second (e.g. 30 FPS)
        do not artificially inflate presence duration.
        """
        tracker = StudentPresenceTracker("STU-3", "CS-03")
        # Record 30 detections at 30 FPS inside the same second (t = 100.0, 100.033, 100.066, ...)
        for i in range(30):
            tracker.record_observation(timestamp=100.0 + (i * 0.033), is_verified_identity=True, confidence=0.9)

        # 30 frame observations in 1 second should yield exactly 1 second (1/60th min)
        self.assertEqual(len(tracker.verified_second_bins), 1)
        self.assertAlmostEqual(tracker.verified_minutes, 1.0 / 60.0, places=4)

    def test_uncertain_identity_exclusion(self):
        """Verify uncertain detections are NOT counted towards verified presence."""
        tracker = StudentPresenceTracker("STU-4", "CS-04")
        # 120 seconds of UNCERTAIN identity
        for sec in range(120):
            tracker.record_observation(timestamp=float(sec), is_verified_identity=False, confidence=0.35)

        self.assertEqual(tracker.verified_minutes, 0.0)
        self.assertEqual(tracker.uncertain_minutes, 2.0)

    def test_teacher_manual_override_with_audit_trail(self):
        """Verify teacher manual review override modifies status and creates audit log."""
        self.engine.register_student("STU-5", "CS-05", "Student Five")
        record = self.engine.calculate_attendance("STU-5")
        self.assertEqual(record.status, AttendanceStatus.BELOW_REQUIRED)

        # Teacher overrides status to PRESENT with valid justification
        success = self.engine.apply_manual_override(
            student_id="STU-5",
            new_status=AttendanceStatus.PRESENT,
            teacher_id="PROF_WILSON",
            reason="Student was physically present; seated in blind spot with verified lab submission."
        )

        self.assertTrue(success)
        updated_record = self.engine.finalized_records["STU-5"]
        self.assertEqual(updated_record.status, AttendanceStatus.PRESENT)
        self.assertTrue(updated_record.manual_override)
        self.assertEqual(len(self.engine.audit_log), 1)
        self.assertEqual(self.engine.audit_log[0].modified_by, "PROF_WILSON")


if __name__ == "__main__":
    unittest.main()
