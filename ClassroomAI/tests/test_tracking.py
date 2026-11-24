"""
Unit tests for ByteTrack student tracking module.
Verifies track persistence, temporary tracking IDs, occlusion handling, and observation gaps.
"""

import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.student_tracker import StudentTracker


class TestStudentTracker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_path = PROJECT_ROOT / "models" / "yolo" / "yolo11n.pt"
        cls.tracker = StudentTracker(model_path=str(model_path), conf_threshold=0.25)

    def setUp(self):
        self.tracker.reset()

    def test_tracker_initialization(self):
        """Verify tracker starts with clean state."""
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertEqual(self.tracker.frame_index, 0)

    def test_tracking_on_sample_video_frames(self):
        """Verify tracking persistence across multiple sequential frames using sample asset."""
        try:
            from ultralytics.utils import ASSETS
            sample_path = ASSETS / "bus.jpg"
            if not sample_path.exists():
                self.skipTest("Sample image not found")

            frame = cv2.imread(str(sample_path))

            # Simulate 5 consecutive video frames with subtle shift
            for i in range(5):
                # Apply slight translation to simulate camera motion
                M = np.float32([[1, 0, i * 2], [0, 1, 0]])
                shifted = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

                result = self.tracker.update(shifted)
                self.assertEqual(result["frame_index"], i + 1)
                self.assertIn("active_tracks", result)
                self.assertIn("total_active_students", result)
                self.assertGreater(result["total_active_students"], 0)

            # Check that tracks were maintained
            self.assertGreater(len(self.tracker.tracks), 0)
            sample_track = next(iter(self.tracker.tracks.values()))
            self.assertGreaterEqual(sample_track.total_observed_frames, 3)
            self.assertFalse(sample_track.is_occluded)
            print(f"\n[Test Tracker] Tracked {len(self.tracker.tracks)} students across 5 frames.")
            print(f"[Test Tracker] Sample Track ID: {sample_track.track_id}, Observed Frames: {sample_track.total_observed_frames}")

        except Exception as e:
            self.fail(f"Tracking sequential frames failed: {e}")

    def test_occlusion_handling(self):
        """Verify that when a student is not detected, unobserved frame count increments and is_occluded becomes True."""
        from ultralytics.utils import ASSETS
        frame = cv2.imread(str(ASSETS / "bus.jpg"))

        # Step 1: Detect and register tracks
        self.tracker.update(frame)
        initial_track_count = len(self.tracker.tracks)
        self.assertGreater(initial_track_count, 0)

        # Step 2: Feed blank frame simulating complete occlusion
        blank_frame = np.zeros_like(frame)
        result = self.tracker.update(blank_frame)

        self.assertEqual(len(result["active_tracks"]), 0)
        self.assertEqual(len(result["occluded_tracks"]), initial_track_count)

        for track in self.tracker.tracks.values():
            self.assertTrue(track.is_occluded)
            self.assertEqual(track.consecutive_unobserved_frames, 1)


if __name__ == "__main__":
    unittest.main()
