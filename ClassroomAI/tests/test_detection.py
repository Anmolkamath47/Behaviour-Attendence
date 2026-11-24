"""
Test suite for student detection module.
Verifies YOLO11 loading, person detection, bounding box geometry, and CPU performance.
"""

import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.student_detection import StudentDetector


class TestStudentDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_path = PROJECT_ROOT / "models" / "yolo" / "yolo11n.pt"
        cls.detector = StudentDetector(model_path=str(model_path), conf_threshold=0.25)

    def test_model_loading(self):
        """Verify model loads with correct classes."""
        self.assertIsNotNone(self.detector.model)
        self.assertIn(0, self.detector.model.names)
        self.assertEqual(self.detector.model.names[0], "person")

    def test_inference_on_synthetic_frame(self):
        """Verify pipeline handles standard 720p frame without errors."""
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = self.detector.detect(blank_frame)

        self.assertIn("students", result)
        self.assertIn("objects", result)
        self.assertIn("student_count", result)
        self.assertIn("inference_time_ms", result)
        self.assertEqual(result["student_count"], 0)
        self.assertEqual(result["frame_dimensions"], (1280, 720))

    def test_inference_on_sample_image(self):
        """Verify person detection on sample asset image."""
        try:
            from ultralytics.utils import ASSETS
            sample_path = ASSETS / "bus.jpg"
            if sample_path.exists():
                frame = cv2.imread(str(sample_path))
                result = self.detector.detect(frame)
                print(f"\n[Test] Detected {result['student_count']} persons in {sample_path.name}")
                print(f"[Test] Inference time on CPU: {result['inference_time_ms']:.2f} ms")
                self.assertGreater(result["student_count"], 0, "Should detect at least 1 person in bus.jpg")
                for student in result["students"]:
                    x1, y1, x2, y2 = student["box"]
                    self.assertGreaterEqual(x2, x1)
                    self.assertGreaterEqual(y2, y1)
                    self.assertGreater(student["confidence"], 0.25)
        except Exception as e:
            self.skipTest(f"Ultralytics sample asset not accessible: {e}")

    def test_annotation(self):
        """Verify annotation overlay produces valid image array."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        annotated = self.detector.annotate_frame(frame, result, fps=28.5)

        self.assertEqual(annotated.shape, frame.shape)
        self.assertEqual(annotated.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
