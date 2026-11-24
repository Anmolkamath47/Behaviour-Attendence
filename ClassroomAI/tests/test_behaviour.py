"""
Unit tests for Behaviour Observation Module:
- TSM temporal shift layer and video sequence buffer
- XGBoost inference on 12-D structured features
- BehaviourEngine fusion and observation timeline accumulators
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.behaviour.tsm_model import TSMBehaviourClassifier, BEHAVIOUR_CLASSES
from src.behaviour.xgboost_classifier import XGBoostBehaviourClassifier
from src.behaviour.behaviour_engine import BehaviourEngine


class TestBehaviourModule(unittest.TestCase):

    def setUp(self):
        self.engine = BehaviourEngine()

    def test_xgboost_classifier_loaded(self):
        """Verify XGBoost classifier loads trained model and predicts probabilities."""
        classifier = XGBoostBehaviourClassifier()
        # Test vector with phone nearby
        feat_vector = np.zeros(12, dtype=np.float32)
        feat_vector[11] = 1.0  # phone nearby
        feat_vector[0] = 20.0  # looking down
        feat_vector[5] = 1.0

        top_class, conf, probs = classifier.predict(feat_vector)
        self.assertIn(top_class, BEHAVIOUR_CLASSES)
        self.assertEqual(len(probs), len(BEHAVIOUR_CLASSES))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=2)

    def test_behaviour_engine_integration(self):
        """Verify multi-modal fusion and cumulative time tracking."""
        dummy_crop = np.full((120, 80, 3), 180, dtype=np.uint8)

        # Feed 10 consecutive frames
        for _ in range(10):
            obs = self.engine.update_student(
                track_id=1,
                student_crop=dummy_crop,
                phone_nearby=False,
                delta_time_sec=0.1
            )

        self.assertEqual(obs.track_id, 1)
        self.assertIn(obs.primary_behaviour, BEHAVIOUR_CLASSES)
        self.assertGreater(obs.confidence, 0.0)
        self.assertTrue(obs.is_observable)

        summary = self.engine.get_summary_for_track(1)
        total_time = sum(summary.values())
        self.assertAlmostEqual(total_time, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
