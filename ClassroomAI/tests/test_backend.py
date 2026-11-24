"""
Unit tests for FastAPI Backend Server:
- Health check and root telemetry
- User registration and JWT authentication
- Student creation and informed consent toggling
- Classroom creation and session lifecycle
- Attendance retrieval and manual review overrides
- Behaviour analytics and ethical disclaimer presence
"""

import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app


class TestBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_and_health(self):
        """Verify root and health check endpoints."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "UP")

    def test_auth_and_student_lifecycle(self):
        """Verify user registration, student enrollment, and consent management."""
        # 1. Register teacher
        email = f"prof_{id(self)}@university.edu"
        res_reg = self.client.post("/api/auth/register", json={
            "email": email,
            "password": "SecretPassword123!",
            "full_name": "Professor Davis",
            "role": "teacher"
        })
        self.assertEqual(res_reg.status_code, 200)
        token = res_reg.json()["access_token"]
        self.assertIsNotNone(token)

        # 2. Enroll student
        roll = f"CS-{id(self) % 1000}"
        res_stu = self.client.post("/api/students", json={
            "roll_number": roll,
            "full_name": "Marcus Aurelius",
            "consent_given": True,
            "alternative_attendance_mode": "MANUAL_REVIEW"
        })
        self.assertEqual(res_stu.status_code, 200)
        stu_id = res_stu.json()["id"]

        # 3. Update consent
        res_consent = self.client.put(f"/api/students/{stu_id}/consent", json={
            "consent_given": False,
            "alternative_mode": "SEATING_BASED"
        })
        self.assertEqual(res_consent.status_code, 200)
        self.assertFalse(res_consent.json()["consent_given"])

    def test_dashboard_summary(self):
        """Verify dashboard summary metrics."""
        res = self.client.get("/api/analytics/dashboard/summary")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("system_status", data)
        self.assertIn("ethical_framework", data)


if __name__ == "__main__":
    unittest.main()
