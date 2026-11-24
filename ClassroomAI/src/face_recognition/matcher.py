"""
ClassroomAI - Face Matcher & Student Consent Registry
Enforces informed consent, matches 512-D ArcFace embeddings using cosine similarity,
flags low-confidence matches as UNKNOWN, and provides alternative attendance workflows
for non-consenting students.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("ClassroomAI.FaceMatcher")

DEFAULT_SIMILARITY_THRESHOLD = 0.55


@dataclass
class StudentProfile:
    """Represents an enrolled student in the classroom database."""
    student_id: str
    roll_number: str
    full_name: str
    consent_given: bool = False
    consent_timestamp: Optional[str] = None
    alternative_attendance_mode: str = "MANUAL_REVIEW"
    embeddings: List[np.ndarray] = field(default_factory=list)
    average_embedding: Optional[np.ndarray] = None

    def add_reference_embedding(self, embedding: np.ndarray):
        """Add an enrolled face embedding and update mean representation."""
        if not self.consent_given:
            logger.warning(
                f"Cannot add biometric embedding for Student {self.roll_number}: "
                f"Informed consent has NOT been granted."
            )
            return

        norm = np.linalg.norm(embedding)
        unit_emb = embedding / norm if norm > 1e-6 else embedding
        self.embeddings.append(unit_emb)

        # Recompute centroid embedding
        all_embs = np.array(self.embeddings)
        mean_emb = np.mean(all_embs, axis=0)
        mean_norm = np.linalg.norm(mean_emb)
        self.average_embedding = mean_emb / mean_norm if mean_norm > 1e-6 else mean_emb


@dataclass
class MatchResult:
    """Outcome of matching an observed face embedding against enrolled students."""
    matched: bool
    student_id: Optional[str]
    roll_number: Optional[str]
    name: Optional[str]
    similarity: float
    status: str  # "VERIFIED_IDENTITY", "UNKNOWN", "NO_CONSENT_ALTERNATIVE"


class FaceMatcher:
    """
    Manages enrolled student biometric identities, enforces consent policies,
    and performs cosine-similarity matching.
    """

    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        self.registry: Dict[str, StudentProfile] = {}  # student_id -> StudentProfile

    def enroll_student(
        self,
        student_id: str,
        roll_number: str,
        full_name: str,
        consent_given: bool = False,
        alternative_mode: str = "MANUAL_REVIEW"
    ) -> StudentProfile:
        """Register a student with explicit consent status."""
        profile = StudentProfile(
            student_id=student_id,
            roll_number=roll_number,
            full_name=full_name,
            consent_given=consent_given,
            consent_timestamp=datetime.utcnow().isoformat() if consent_given else None,
            alternative_attendance_mode=alternative_mode
        )
        self.registry[student_id] = profile
        logger.info(
            f"Enrolled student: {full_name} ({roll_number}) | "
            f"Consent: {'GRANTED' if consent_given else 'REFUSED/PENDING'} | "
            f"Alternative mode: {alternative_mode}"
        )
        return profile

    def update_consent(self, student_id: str, consent_given: bool) -> bool:
        """Update consent and purge biometric embeddings if consent is revoked."""
        if student_id not in self.registry:
            return False

        profile = self.registry[student_id]
        profile.consent_given = consent_given
        if consent_given:
            profile.consent_timestamp = datetime.utcnow().isoformat()
            logger.info(f"Consent GRANTED for student: {profile.roll_number}")
        else:
            profile.consent_timestamp = None
            # Purge biometric data immediately upon consent withdrawal
            profile.embeddings.clear()
            profile.average_embedding = None
            logger.info(f"Consent REVOKED for student: {profile.roll_number}. Biometric embeddings purged.")
        return True

    def match(self, query_embedding: np.ndarray) -> MatchResult:
        """
        Compare query embedding against all consented enrolled students.

        Returns:
            MatchResult. If best similarity < threshold, status is UNKNOWN.
            Never forces an identity match.
        """
        # Ensure unit norm
        norm = np.linalg.norm(query_embedding)
        if norm > 1e-6:
            query = query_embedding / norm
        else:
            return MatchResult(False, None, None, "Unknown", 0.0, "UNKNOWN")

        best_sim = -1.0
        best_profile: Optional[StudentProfile] = None

        # Compare only against students who gave informed consent and have registered embeddings
        for profile in self.registry.values():
            if not profile.consent_given or profile.average_embedding is None:
                continue

            sim = float(np.dot(query, profile.average_embedding))
            if sim > best_sim:
                best_sim = sim
                best_profile = profile

        # Check threshold
        if best_profile is not None and best_sim >= self.similarity_threshold:
            return MatchResult(
                matched=True,
                student_id=best_profile.student_id,
                roll_number=best_profile.roll_number,
                name=best_profile.full_name,
                similarity=round(best_sim, 3),
                status="VERIFIED_IDENTITY"
            )

        # Fallback to UNKNOWN when confidence is insufficient
        return MatchResult(
            matched=False,
            student_id=None,
            roll_number=None,
            name="Unknown",
            similarity=round(max(0.0, best_sim), 3),
            status="UNKNOWN"
        )

    def get_non_consenting_students(self) -> List[StudentProfile]:
        """Retrieve all students requiring non-biometric alternative attendance."""
        return [p for p in self.registry.values() if not p.consent_given]
