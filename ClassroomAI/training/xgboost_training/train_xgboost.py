"""
ClassroomAI - XGBoost Model Training Pipeline
Trains an XGBClassifier on multi-modal structured classroom features.
Evaluates Precision, Recall, F1 score, and Confusion Matrix, and saves the trained model.
"""

import os
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.behaviour.tsm_model import BEHAVIOUR_CLASSES
from src.pose.feature_extractor import FEATURE_NAMES


def generate_synthetic_classroom_dataset(num_samples_per_class: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize realistic classroom feature distributions conforming to the 8 activity classes
    for baseline model training and validation.
    """
    X_list = []
    y_list = []

    np.random.seed(42)

    for class_idx, class_name in enumerate(BEHAVIOUR_CLASSES):
        for _ in range(num_samples_per_class):
            # Base features:
            # [pitch, yaw, roll, shoulder_tilt, torso_lean, is_look_down, is_facing_front, is_look_away, hand_raised, hands_desk, hands_face, phone_nearby]
            pitch = np.random.normal(0, 5)
            yaw = np.random.normal(0, 8)
            roll = np.random.normal(0, 4)
            shoulder_tilt = np.random.normal(0, 4)
            torso_lean = np.random.uniform(0, 10)
            is_look_down = 0.0
            is_facing_front = 1.0
            is_look_away = 0.0
            hand_raised = 0.0
            hands_desk = 0.0
            hands_face = 0.0
            phone_nearby = 0.0

            if class_name == "facing_teaching_area":
                is_facing_front = 1.0
            elif class_name == "writing":
                pitch = np.random.uniform(15, 35)
                is_look_down = 1.0
                is_facing_front = 0.0
                hands_desk = 1.0
                torso_lean = np.random.uniform(10, 25)
            elif class_name == "raising_hand":
                hand_raised = np.random.uniform(0.6, 1.0)
            elif class_name == "using_mobile_phone":
                phone_nearby = 1.0
                pitch = np.random.uniform(18, 35)
                is_look_down = 1.0
                is_facing_front = 0.0
            elif class_name == "head_down_posture":
                pitch = np.random.uniform(30, 45)
                is_look_down = 1.0
                is_facing_front = 0.0
                torso_lean = np.random.uniform(20, 35)
            elif class_name == "looking_away":
                yaw = np.random.choice([np.random.uniform(30, 60), np.random.uniform(-60, -30)])
                is_look_away = 1.0
                is_facing_front = 0.0
            elif class_name == "head_resting_on_hand":
                hands_face = 1.0
                shoulder_tilt = np.random.choice([-15.0, 15.0])
            elif class_name == "other_or_unknown":
                is_facing_front = 0.0
                is_look_away = 0.5

            feat = np.array([
                pitch, yaw, roll, shoulder_tilt, torso_lean,
                is_look_down, is_facing_front, is_look_away,
                hand_raised, hands_desk, hands_face, phone_nearby
            ], dtype=np.float32)

            X_list.append(feat)
            y_list.append(class_idx)

    return np.array(X_list), np.array(y_list)


def train_and_export_xgboost():
    print("Generating classroom training dataset...")
    X, y = generate_synthetic_classroom_dataset(num_samples_per_class=200)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset split: {len(X_train)} training samples, {len(X_val)} validation samples.")

    # Train XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(BEHAVIOUR_CLASSES),
        random_state=42
    )

    print("Training XGBoost behaviour classifier...")
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_val)
    print("\n--- Model Evaluation Report ---")
    print(classification_report(y_val, preds, target_names=BEHAVIOUR_CLASSES))

    # Save trained model
    out_dir = PROJECT_ROOT / "models" / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "xgboost_behaviour.json"
    model.save_model(str(out_file))
    print(f"Successfully trained and saved XGBoost model to: {out_file}")


if __name__ == "__main__":
    train_and_export_xgboost()
