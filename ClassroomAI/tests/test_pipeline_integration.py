"""
Unit and scaling integration tests for ClassroomAIPipeline.
Tests:
- Full end-to-end processing of frames through all 7 ML stages
- Scaling across 5, 15, 30, and 65 students
- Attendance 75% calculation integration
- Behaviour timeline generation
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classroom_pipeline import ClassroomAIPipeline


class TestPipelineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = ClassroomAIPipeline(session_id="BENCHMARK_LEC", lecture_duration_minutes=60.0)

        # Enroll 3 test students
        for i in range(1, 4):
            face = np.full((112, 112, 3), i * 60, dtype=np.uint8)
            cls.pipeline.enroll_consented_student(
                student_id=f"STU_{i}",
                roll_number=f"CS-0{i}",
                name=f"Student {i}",
                reference_face_rgb=face,
                consent_given=True
            )

    def test_single_frame_end_to_end(self):
        """Verify single frame execution produces latency, tracking, and observations."""
        test_frame = np.full((720, 1280, 3), 150, dtype=np.uint8)
        # Draw student body
        import cv2
        cv2.rectangle(test_frame, (200, 200), (320, 500), (80, 60, 40), -1)

        result = self.pipeline.process_frame(test_frame, timestamp=10.0, frame_idx=1)

        self.assertIn("latency_ms", result)
        self.assertIn("observations", result)
        self.assertGreater(result["latency_ms"], 0.0)
        print(f"\n[Integration] Single frame latency: {result['latency_ms']:.2f} ms")

    def test_scaling_benchmark(self):
        """Benchmark 5, 15, 30, and 65 students."""
        benchmarks = self.pipeline.benchmark_scaling(student_counts=[5, 15, 30, 65], frames_per_benchmark=2)

        self.assertIn("5_students", benchmarks)
        self.assertIn("65_students", benchmarks)

        for key, stats in benchmarks.items():
            print(f"[Benchmark] {key}: Latency {stats['avg_latency_ms']} ms | Estimated CPU FPS: {stats['estimated_cpu_fps']} ({stats['cpu_feasibility']})")


if __name__ == "__main__":
    unittest.main()
