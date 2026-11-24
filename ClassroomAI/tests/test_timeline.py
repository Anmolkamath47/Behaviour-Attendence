"""
Unit tests for Behaviour Timeline & Analytics Module.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.behaviour.timeline_analytics import StudentObservationTimeline, ETHICAL_DISCLAIMER


class TestTimelineAnalytics(unittest.TestCase):

    def test_timeline_accumulation_and_buckets(self):
        """Verify continuous activity segments and 5-minute bucket formatting."""
        timeline = StudentObservationTimeline("S-100", "CS-10", "LEC-01")

        # Simulate 10 minutes (600 seconds) of writing
        for sec in range(0, 600, 2):
            timeline.record_activity_sample(
                timestamp=float(sec),
                activity="writing",
                confidence=0.88,
                is_observable=True
            )

        # Simulate 5 minutes (300 seconds) of facing teaching area
        for sec in range(600, 900, 2):
            timeline.record_activity_sample(
                timestamp=float(sec),
                activity="facing_teaching_area",
                confidence=0.91,
                is_observable=True
            )

        breakdown = timeline.get_activity_breakdown()
        self.assertAlmostEqual(breakdown["writing"], 10.0, delta=0.5)
        self.assertAlmostEqual(breakdown["facing_teaching_area"], 5.0, delta=0.5)

        # Test Recharts 5-minute buckets
        buckets = timeline.generate_timeline_buckets(bucket_minutes=5.0)
        self.assertEqual(len(buckets), 3)  # 0-5m, 5-10m, 10-15m
        self.assertIn("interval", buckets[0])
        self.assertIn("writing", buckets[0])

    def test_baseline_comparison(self):
        """Verify baseline comparison calculation and presence of ethical disclaimer."""
        timeline = StudentObservationTimeline("S-101", "CS-11", "LEC-01")
        for sec in range(0, 1800, 2):  # 30 min of writing
            timeline.record_activity_sample(float(sec), "writing", 0.85)

        historical = {"writing": 40.0, "facing_teaching_area": 50.0}
        comparison = timeline.compare_with_baseline(historical)

        self.assertIn("percentage_deltas", comparison)
        self.assertIn("ethical_disclaimer", comparison)
        self.assertEqual(comparison["ethical_disclaimer"], ETHICAL_DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
