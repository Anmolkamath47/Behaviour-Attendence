"""
ClassroomAI - XGBoost Behaviour Classifier
Classifies structured multi-modal features (head pose, body posture, and object context)
into the 8 classroom activity classes. Acts as a complementary classifier to TSM.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import xgboost as xgb

from .tsm_model import BEHAVIOUR_CLASSES

logger = logging.getLogger("ClassroomAI.XGBoost")


class XGBoostBehaviourClassifier:
    """
    XGBoost classifier operating on 12-dimensional structured features.
    Provides fast, deterministic tabular classification on CPU.
    """

    def __init__(self, model_path: Optional[str] = None):
        project_root = Path(__file__).resolve().parent.parent.parent
        self.default_model_path = project_root / "models" / "xgboost" / "xgboost_behaviour.json"
        self.default_model_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = None
        target_path = Path(model_path) if model_path else self.default_model_path

        if target_path.exists():
            try:
                self.model = xgb.Booster()
                self.model.load_model(str(target_path))
                logger.info(f"Loaded trained XGBoost model from {target_path}")
            except Exception as e:
                logger.warning(f"Could not load XGBoost model: {e}")

        if self.model is None:
            # Fall back to heuristic rule-based classifier until trained model is saved
            logger.info("Operating in heuristic rule-based mode until XGBoost weights are trained.")

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predicts behaviour class from 12-D feature vector.

        Returns:
            Tuple of (class_name, confidence, probabilities_array)
        """
        assert len(feature_vector) == 12, f"Expected 12-D feature vector, got {len(feature_vector)}"

        if self.model is not None:
            dmatrix = xgb.DMatrix(np.expand_dims(feature_vector, axis=0))
            probs = self.model.predict(dmatrix)[0]
            top_idx = int(np.argmax(probs))
            return (BEHAVIOUR_CLASSES[top_idx], float(probs[top_idx]), probs)

        # Heuristic rules conforming to ethical and physical definitions:
        # [0:pitch, 1:yaw, 2:roll, 3:shoulder_tilt, 4:torso_lean, 5:look_down,
        #  6:facing_front, 7:looking_away, 8:hand_raised_conf, 9:hands_desk,
        #  10:hands_face, 11:phone_nearby]
        probs = np.zeros(len(BEHAVIOUR_CLASSES), dtype=np.float32)

        # Hand raised
        if feature_vector[8] > 0.4:
            probs[2] = 0.85
        # Phone nearby
        elif feature_vector[11] > 0.5:
            probs[3] = 0.80
        # Hands near face
        elif feature_vector[10] > 0.5:
            probs[6] = 0.75
        # Hands on desk + head looking down -> Writing
        elif feature_vector[9] > 0.5 and feature_vector[5] > 0.5:
            probs[1] = 0.78
        # Head down resting (pitch very low, no hands on desk)
        elif feature_vector[5] > 0.5 and feature_vector[0] > 25.0:
            probs[4] = 0.70
        # Looking away
        elif feature_vector[7] > 0.5:
            probs[5] = 0.72
        # Facing front
        elif feature_vector[6] > 0.5:
            probs[0] = 0.80
        else:
            probs[7] = 0.60

        # Normalize
        s = np.sum(probs)
        if s > 0:
            probs /= s
        top_idx = int(np.argmax(probs))
        return (BEHAVIOUR_CLASSES[top_idx], round(float(probs[top_idx]), 3), probs)
