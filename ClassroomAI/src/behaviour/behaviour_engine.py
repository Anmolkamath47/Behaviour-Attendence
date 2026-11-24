"""
ClassroomAI - Multi-Model Behaviour Fusion Engine
Blends TSM temporal video sequence predictions with XGBoost structured classification.
Enforces that behaviour observations are experimental estimates, never psychological measurements.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .tsm_model import TSMBehaviourClassifier, BEHAVIOUR_CLASSES
from .xgboost_classifier import XGBoostBehaviourClassifier
from ..pose.feature_extractor import ClassroomFeatureExtractor

logger = logging.getLogger("ClassroomAI.BehaviourEngine")


@dataclass
class BehaviourObservation:
    """Represents a student's observable visible classroom behaviour at a given moment."""
    track_id: int
    primary_behaviour: str
    confidence: float
    tsm_prediction: str
    xgboost_prediction: str
    is_observable: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    disclaimer: str = "Experimental observable activity estimate; does not measure mental attention or intelligence."


class BehaviourEngine:
    """
    Synchronizes TSM temporal sequence analysis and XGBoost structured classification
    for tracked classroom students.
    """

    def __init__(self, tsm_weight: float = 0.45, xgb_weight: float = 0.55):
        self.tsm_weight = tsm_weight
        self.xgb_weight = xgb_weight

        self.feature_extractor = ClassroomFeatureExtractor()
        self.tsm_classifier = TSMBehaviourClassifier()
        self.xgb_classifier = XGBoostBehaviourClassifier()

        # Cumulative observation durations per track_id:
        # track_id -> {behaviour_class: total_observed_seconds}
        self.behaviour_durations: Dict[int, Dict[str, float]] = {}

    def update_student(
        self,
        track_id: int,
        student_crop: np.ndarray,
        phone_nearby: bool = False,
        delta_time_sec: float = 0.033
    ) -> BehaviourObservation:
        """
        Updates behaviour observation for a tracked student.

        Args:
            track_id: Temporary student track ID
            student_crop: BGR student image crop
            phone_nearby: Contextual flag if mobile phone is detected near the student
            delta_time_sec: Frame time interval in seconds (e.g. 1/30 = 0.033)

        Returns:
            BehaviourObservation instance
        """
        if student_crop is None or student_crop.size == 0:
            return BehaviourObservation(
                track_id=track_id,
                primary_behaviour="other_or_unknown",
                confidence=0.0,
                tsm_prediction="other_or_unknown",
                xgboost_prediction="other_or_unknown",
                is_observable=False
            )

        # 1. Update TSM video sequence buffer
        self.tsm_classifier.push_frame(track_id, student_crop)
        tsm_class, tsm_conf, tsm_probs = self.tsm_classifier.predict_behaviour(track_id)

        # 2. Extract structured features for XGBoost
        features = self.feature_extractor.extract_features(student_crop, phone_nearby=phone_nearby)
        xgb_class, xgb_conf, xgb_probs = self.xgb_classifier.predict(features["vector"])

        # 3. Ensemble multi-modal probabilities
        blended_probs = (self.tsm_weight * tsm_probs) + (self.xgb_weight * xgb_probs)
        top_idx = int(np.argmax(blended_probs))
        primary_behaviour = BEHAVIOUR_CLASSES[top_idx]
        confidence = round(float(blended_probs[top_idx]), 3)

        # 4. Accumulate observation durations
        if track_id not in self.behaviour_durations:
            self.behaviour_durations[track_id] = {c: 0.0 for c in BEHAVIOUR_CLASSES}
        self.behaviour_durations[track_id][primary_behaviour] += delta_time_sec

        return BehaviourObservation(
            track_id=track_id,
            primary_behaviour=primary_behaviour,
            confidence=confidence,
            tsm_prediction=tsm_class,
            xgboost_prediction=xgb_class,
            is_observable=True,
            details={
                "head_direction": features["head_orientation"].direction,
                "pitch": features["head_orientation"].pitch,
                "yaw": features["head_orientation"].yaw,
                "hand_raised": features["posture"].hand_raised,
                "hands_near_desk": features["posture"].hands_near_desk,
                "hands_near_face": features["posture"].hands_near_face
            }
        )

    def get_summary_for_track(self, track_id: int) -> Dict[str, float]:
        """Return cumulative observation duration in seconds for each activity class."""
        return self.behaviour_durations.get(track_id, {c: 0.0 for c in BEHAVIOUR_CLASSES})
