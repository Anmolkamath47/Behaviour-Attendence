"""
Unit tests for Pose, Head Orientation, and Feature Extractor Module.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose.head_pose import HeadPoseEstimator
from src.pose.mediapipe_analyzer import PostureAnalyzer
from src.pose.feature_extractor import ClassroomFeatureExtractor, FEATURE_NAMES


class TestPoseModule(unittest.TestCase):

    def setUp(self):
        self.extractor = ClassroomFeatureExtractor()

    def test_head_pose_estimation(self):
        """Verify continuous Euler angle extraction and discrete direction."""
        estimator = HeadPoseEstimator()
        dummy_crop = np.full((100, 80, 3), 150, dtype=np.uint8)
        head_orient = estimator.estimate_from_crop(dummy_crop)

        self.assertIn(head_orient.direction, [
            "FACING_FORWARD", "LOOKING_DOWN", "LOOKING_LEFT", "LOOKING_RIGHT", "LOOKING_UP"
        ])
        self.assertIsInstance(head_orient.pitch, float)
        self.assertIsInstance(head_orient.yaw, float)

    def test_feature_extractor_vector_length(self):
        """Verify feature extractor generates a 12-dimensional vector."""
        dummy_student = np.full((150, 100, 3), 120, dtype=np.uint8)
        res = self.extractor.extract_features(dummy_student, phone_nearby=True)

        self.assertEqual(len(res["vector"]), 12)
        self.assertEqual(len(res["feature_names"]), 12)
        self.assertEqual(res["feature_names"], FEATURE_NAMES)
        self.assertEqual(res["vector"][-1], 1.0)  # phone_detected_nearby == True


if __name__ == "__main__":
    unittest.main()
