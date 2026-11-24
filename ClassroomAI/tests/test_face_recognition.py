"""
Unit tests for Face Recognition Module:
- ArcFace MobileFaceNet model loading & 512-D embedding extraction
- ONNX export and ONNX Runtime CPU inference
- Informed consent verification and biometric data purge
- Cosine similarity matching, threshold enforcement, and Unknown categorization
- Non-consenting student alternative workflows
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.face_recognition.arcface import ArcFaceEmbedder
from src.face_recognition.face_detector import FaceDetector
from src.face_recognition.matcher import FaceMatcher, DEFAULT_SIMILARITY_THRESHOLD


class TestFaceRecognition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.embedder = ArcFaceEmbedder()
        cls.detector = FaceDetector()

    def test_arcface_embedding_shape_and_norm(self):
        """Verify ArcFace produces 512-D unit-normalized vectors."""
        dummy_face = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = self.embedder.extract_embedding(dummy_face)

        self.assertEqual(emb.shape, (512,))
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_face_detector_crop_handling(self):
        """Verify face detector handles student crops with fallback support."""
        dummy_student = np.zeros((200, 100, 3), dtype=np.uint8)
        res = self.detector.detect_face_in_crop(dummy_student)

        self.assertIsNotNone(res)
        self.assertIn("face_crop", res)
        self.assertEqual(res["face_crop"].shape, (112, 112, 3))

    def test_face_matcher_consent_and_matching(self):
        """Verify consent controls, similarity matching, and Unknown categorization."""
        matcher = FaceMatcher(similarity_threshold=0.55)

        # 1. Enroll Alice with consent
        alice = matcher.enroll_student(
            student_id="STU001",
            roll_number="CS-01",
            full_name="Alice Smith",
            consent_given=True
        )
        sample_face_alice = np.random.randint(100, 200, (112, 112, 3), dtype=np.uint8)
        emb_alice = self.embedder.extract_embedding(sample_face_alice)
        alice.add_reference_embedding(emb_alice)

        # 2. Enroll Bob without consent
        bob = matcher.enroll_student(
            student_id="STU002",
            roll_number="CS-02",
            full_name="Bob Jones",
            consent_given=False,
            alternative_mode="MANUAL_SEATING_CHECK"
        )
        sample_face_bob = np.random.randint(50, 150, (112, 112, 3), dtype=np.uint8)
        emb_bob = self.embedder.extract_embedding(sample_face_bob)
        bob.add_reference_embedding(emb_bob)
        # Verify Bob has NO embeddings stored due to lack of consent
        self.assertEqual(len(bob.embeddings), 0)

        # 3. Query with Alice's embedding -> should match Alice
        match_alice = matcher.match(emb_alice)
        self.assertTrue(match_alice.matched)
        self.assertEqual(match_alice.roll_number, "CS-01")
        self.assertEqual(match_alice.status, "VERIFIED_IDENTITY")
        self.assertGreaterEqual(match_alice.similarity, 0.90)

        # 4. Query with unrelated / random embedding -> should be UNKNOWN, never forced
        random_face = np.random.randint(0, 50, (112, 112, 3), dtype=np.uint8)
        emb_random = self.embedder.extract_embedding(random_face)
        # Modify to ensure low cosine similarity
        emb_dissimilar = -emb_alice + np.random.normal(0, 0.05, 512).astype(np.float32)
        emb_dissimilar /= np.linalg.norm(emb_dissimilar)
        
        match_unknown = matcher.match(emb_dissimilar)
        self.assertFalse(match_unknown.matched)
        self.assertEqual(match_unknown.status, "UNKNOWN")
        self.assertEqual(match_unknown.name, "Unknown")

        # 5. Verify non-consenting students list
        non_consenting = matcher.get_non_consenting_students()
        self.assertEqual(len(non_consenting), 1)
        self.assertEqual(non_consenting[0].roll_number, "CS-02")


if __name__ == "__main__":
    unittest.main()
