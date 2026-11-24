"""
ClassroomAI - Physical Presence Tracker
Records time-binned verified presence observations per student.
Avoids double counting and strictly excludes uncertain identity matches from verified presence.
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

logger = logging.getLogger("ClassroomAI.PresenceTracker")


@dataclass
class PresenceObservation:
    """Represents a discrete presence observation event."""
    timestamp: float
    time_bin_second: int  # Integer second epoch to prevent double counting
    is_verified: bool     # True ONLY if identity is verified with high confidence
    confidence: float
    status: str           # "VERIFIED_PRESENT", "UNCERTAIN_IDENTITY", "OCCLUDED"


class StudentPresenceTracker:
    """
    Tracks reliable presence intervals for an individual student.
    Uses discrete second-level bins to guarantee non-overlapping presence accumulation.
    """

    def __init__(self, student_id: str, roll_number: str):
        self.student_id = student_id
        self.roll_number = roll_number

        # Set of verified seconds (prevents double-counting from 30 FPS multiple detections)
        self.verified_second_bins: Set[int] = set()
        # Set of uncertain seconds
        self.uncertain_second_bins: Set[int] = set()

        self.first_seen_timestamp: Optional[float] = None
        self.last_seen_timestamp: Optional[float] = None

    def record_observation(
        self,
        timestamp: float,
        is_verified_identity: bool,
        confidence: float
    ):
        """
        Record a camera observation for this student.

        Args:
            timestamp: Epoch seconds of frame
            is_verified_identity: True only if identity match exceeded threshold
            confidence: Match confidence score
        """
        sec_bin = int(timestamp)

        if self.first_seen_timestamp is None or timestamp < self.first_seen_timestamp:
            self.first_seen_timestamp = timestamp
        if self.last_seen_timestamp is None or timestamp > self.last_seen_timestamp:
            self.last_seen_timestamp = timestamp

        if is_verified_identity:
            self.verified_second_bins.add(sec_bin)
            # Remove from uncertain if previously uncertain in this same second
            self.uncertain_second_bins.discard(sec_bin)
        else:
            if sec_bin not in self.verified_second_bins:
                self.uncertain_second_bins.add(sec_bin)

    @property
    def verified_minutes(self) -> float:
        """Total verified physical presence in minutes."""
        return len(self.verified_second_bins) / 60.0

    @property
    def uncertain_minutes(self) -> float:
        """Total unverified / uncertain presence in minutes."""
        return len(self.uncertain_second_bins) / 60.0

    @property
    def arrival_time_str(self) -> Optional[str]:
        if self.first_seen_timestamp:
            return datetime.fromtimestamp(self.first_seen_timestamp).strftime("%H:%M:%S")
        return None

    @property
    def departure_time_str(self) -> Optional[str]:
        if self.last_seen_timestamp:
            return datetime.fromtimestamp(self.last_seen_timestamp).strftime("%H:%M:%S")
        return None
