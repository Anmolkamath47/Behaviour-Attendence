"""
ClassroomAI - Student Behaviour Observation Timeline & Analytics
Generates individual student observation timelines, multi-lecture baseline comparisons,
and aggregates activity time slices.
Enforces the mandatory ethical rule that observations are experimental visual estimates,
not psychological measures of mental attention.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from .tsm_model import BEHAVIOUR_CLASSES

logger = logging.getLogger("ClassroomAI.Timeline")

ETHICAL_DISCLAIMER = (
    "DISCLAIMER: All behaviour metrics are experimental estimates derived solely from observable "
    "physical cues (body orientation, head direction, and hand movement). They do NOT measure mental "
    "attention, comprehension, cognitive focus, interest, or intelligence."
)


@dataclass
class TimelineSegment:
    """Discrete time interval of observable classroom activity."""
    start_second: float
    end_second: float
    duration_seconds: float
    activity: str
    confidence: float
    is_observable: bool = True


class StudentObservationTimeline:
    """
    Constructs an event timeline and statistical profile of a student's observable activities.
    Supports longitudinal baseline comparison across consented lectures.
    """

    def __init__(self, student_id: str, roll_number: str, session_id: str):
        self.student_id = student_id
        self.roll_number = roll_number
        self.session_id = session_id

        self.segments: List[TimelineSegment] = []
        self.current_segment: Optional[TimelineSegment] = None

    def record_activity_sample(
        self,
        timestamp: float,
        activity: str,
        confidence: float,
        is_observable: bool = True
    ):
        """Append or extend timeline segments based on consecutive identical activity."""
        if self.current_segment is None:
            self.current_segment = TimelineSegment(
                start_second=timestamp,
                end_second=timestamp + 1.0,
                duration_seconds=1.0,
                activity=activity,
                confidence=confidence,
                is_observable=is_observable
            )
            return

        # If same activity continues, extend current segment
        if (self.current_segment.activity == activity and
                self.current_segment.is_observable == is_observable and
                timestamp - self.current_segment.end_second <= 2.0):
            self.current_segment.end_second = timestamp + 1.0
            self.current_segment.duration_seconds = self.current_segment.end_second - self.current_segment.start_second
            # Running average confidence
            self.current_segment.confidence = round((self.current_segment.confidence + confidence) / 2.0, 3)
        else:
            # Finalize previous segment and start new one
            self.segments.append(self.current_segment)
            self.current_segment = TimelineSegment(
                start_second=timestamp,
                end_second=timestamp + 1.0,
                duration_seconds=1.0,
                activity=activity,
                confidence=confidence,
                is_observable=is_observable
            )

    def finalize(self):
        """Seal the current active segment."""
        if self.current_segment is not None:
            self.segments.append(self.current_segment)
            self.current_segment = None

    def get_activity_breakdown(self) -> Dict[str, float]:
        """Total duration in minutes spent in each activity class."""
        self.finalize()
        totals = {c: 0.0 for c in BEHAVIOUR_CLASSES}
        totals["unobservable"] = 0.0

        for seg in self.segments:
            duration_min = seg.duration_seconds / 60.0
            if not seg.is_observable:
                totals["unobservable"] += duration_min
            elif seg.activity in totals:
                totals[seg.activity] += duration_min
            else:
                totals["other_or_unknown"] += duration_min

        return {k: round(v, 2) for k, v in totals.items()}

    def generate_timeline_buckets(self, bucket_minutes: float = 5.0) -> List[Dict[str, Any]]:
        """
        Groups timeline data into discrete intervals (e.g. 5-minute blocks)
        for front-end Recharts visualization.
        """
        self.finalize()
        if not self.segments:
            return []

        max_sec = max(seg.end_second for seg in self.segments)
        bucket_sec = bucket_minutes * 60.0
        num_buckets = int(np.ceil(max_sec / bucket_sec)) or 1

        buckets = []
        for b in range(num_buckets):
            b_start = b * bucket_sec
            b_end = (b + 1) * bucket_sec
            b_name = f"{int(b_start // 60)}-{int(b_end // 60)}m"

            b_data = {c: 0.0 for c in BEHAVIOUR_CLASSES}
            b_data["interval"] = b_name

            for seg in self.segments:
                # Calculate overlap between segment and bucket
                overlap_start = max(b_start, seg.start_second)
                overlap_end = min(b_end, seg.end_second)
                if overlap_end > overlap_start:
                    dur_sec = overlap_end - overlap_start
                    if seg.activity in b_data:
                        b_data[seg.activity] += round(dur_sec, 1)

            buckets.append(b_data)

        return buckets

    def compare_with_baseline(
        self,
        historical_baseline: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Compares today's lecture activity distribution against the student's historical baseline.
        Highlights notable differences in observable patterns (e.g. more writing, less front-facing).
        """
        current = self.get_activity_breakdown()
        total_current_time = sum(current.values()) or 1.0

        current_pct = {k: (v / total_current_time) * 100.0 for k, v in current.items()}
        deltas = {}

        for k in BEHAVIOUR_CLASSES:
            base_pct = historical_baseline.get(k, 0.0)
            cur_pct = current_pct.get(k, 0.0)
            deltas[k] = round(cur_pct - base_pct, 1)

        return {
            "current_breakdown_minutes": current,
            "current_percentages": {k: round(v, 1) for k, v in current_pct.items()},
            "historical_baseline_percentages": historical_baseline,
            "percentage_deltas": deltas,
            "ethical_disclaimer": ETHICAL_DISCLAIMER
        }
