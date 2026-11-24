"""
ClassroomAI - Temporal Shift Module (TSM)
Analyzes 8-frame short video sequences of tracked students rather than isolated frames.
Implements zero-FLOP temporal shifts across channels for temporal video action recognition.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("ClassroomAI.TSM")

BEHAVIOUR_CLASSES = [
    "facing_teaching_area",   # 0: Upright, head oriented toward front
    "writing",                # 1: Head inclined, hands active near desk
    "raising_hand",           # 2: Hand/arm extended upward
    "using_mobile_phone",     # 3: Phone detected or hands focused downward on screen
    "head_down_posture",      # 4: Head down resting on desk
    "looking_away",           # 5: Head turned significantly to left or right
    "head_resting_on_hand",   # 6: Hand supporting chin/cheek
    "other_or_unknown"        # 7: Occluded or unclassified visible activity
]


class TemporalShift(nn.Module):
    """
    Temporal Shift Module (TSM) layer.
    Shifts 1/8 of feature channels forward in time and 1/8 backward in time.
    Enables temporal feature exchange with zero extra parameters and zero extra FLOPs.
    """
    def __init__(self, net: nn.Module, n_segment: int = 8, fold_div: int = 8):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N * T, C, H, W)
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)

        fold = c // self.fold_div
        out = torch.zeros_like(x)
        # Shift 1: backward in time
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # Shift 2: forward in time
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
        # Remainder: unchanged
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        out = out.view(nt, c, h, w)
        return self.net(out)


class TSMBackbone(nn.Module):
    """
    Lightweight video sequence backbone for 8-frame student crops.
    Outputs class logits for the 8 classroom behaviour classes.
    """
    def __init__(self, num_classes: int = len(BEHAVIOUR_CLASSES), num_segments: int = 8):
        super().__init__()
        self.num_segments = num_segments
        self.num_classes = num_classes

        # Spatial feature extractor
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Apply temporal shift to intermediate feature representations
        self.conv2 = TemporalShift(
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            n_segment=num_segments
        )
        self.conv3 = TemporalShift(
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            n_segment=num_segments
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, 3, 112, 112)
        n, t, c, h, w = x.size()
        x = x.view(n * t, c, h, w)
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)  # (N*T, 128)
        logits = self.fc(feat)         # (N*T, num_classes)

        # Average logits across temporal segments
        logits = logits.view(n, t, self.num_classes)
        return torch.mean(logits, dim=1)  # (N, num_classes)


class TSMBehaviourClassifier:
    """
    Manages temporal video buffer of recent student crops (8 frames)
    and predicts temporal classroom behaviour.
    """

    def __init__(self, num_segments: int = 8, device: str = "cpu"):
        self.num_segments = num_segments
        self.device = device
        self.model = TSMBackbone(num_classes=len(BEHAVIOUR_CLASSES), num_segments=num_segments)
        self.model.to(self.device)
        self.model.eval()

        # Track buffers: track_id -> List of 8 resized RGB crops
        self.buffers: Dict[int, List[np.ndarray]] = {}

    def push_frame(self, track_id: int, crop: np.ndarray):
        """Append latest crop of student to track temporal sequence buffer."""
        if crop is None or crop.size == 0:
            return
        resized = cv2.resize(crop, (112, 112))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if track_id not in self.buffers:
            self.buffers[track_id] = []

        self.buffers[track_id].append(rgb)
        if len(self.buffers[track_id]) > self.num_segments:
            self.buffers[track_id].pop(0)

    def predict_behaviour(self, track_id: int) -> Tuple[str, float, np.ndarray]:
        """
        Classifies behaviour over the last 8 frames for a given student track.

        Returns:
            Tuple of (class_name, confidence, probabilities_array)
        """
        if track_id not in self.buffers or len(self.buffers[track_id]) < 4:
            # Insufficient temporal history: default to other_or_unknown with 0 confidence
            probs = np.zeros(len(BEHAVIOUR_CLASSES), dtype=np.float32)
            probs[-1] = 1.0
            return ("other_or_unknown", 0.0, probs)

        frames = self.buffers[track_id]
        # Pad up to 8 if needed
        while len(frames) < self.num_segments:
            frames.append(frames[-1])

        # Preprocess: (1, 8, 3, 112, 112)
        arr = np.array(frames, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (0, 3, 1, 2))  # (8, 3, 112, 112)
        tensor_in = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1, 8, 3, 112, 112)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        return (BEHAVIOUR_CLASSES[top_idx], round(top_conf, 3), probs)
