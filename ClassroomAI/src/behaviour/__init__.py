"""
Behaviour Observation Module for ClassroomAI
Integrates TSM (Temporal Shift Module) for video clip sequences and XGBoost for structured features.
Supports the 8 standard classroom activity classes.
"""

from .tsm_model import TemporalShift, TSMBehaviourClassifier, BEHAVIOUR_CLASSES
from .xgboost_classifier import XGBoostBehaviourClassifier
from .behaviour_engine import BehaviourEngine, BehaviourObservation

__all__ = [
    "TemporalShift",
    "TSMBehaviourClassifier",
    "XGBoostBehaviourClassifier",
    "BehaviourEngine",
    "BehaviourObservation",
    "BEHAVIOUR_CLASSES"
]
