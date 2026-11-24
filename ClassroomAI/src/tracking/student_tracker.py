"""
ClassroomAI - ByteTrack Student Tracking Module
Maintains temporary student tracking IDs across video frames, handles occlusion,
records observation gaps, and calculates tracking trajectories.
Adheres strictly to the architectural requirement that tracking IDs are temporary
and must not be treated as permanent student identities.
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Configure logging
logger = logging.getLogger("ClassroomAI.Tracking")

# Colors for professional visualization
COLOR_ACTIVE_TRACK = (235, 99, 37)     # Royal Blue (active)
COLOR_OCCLUDED_TRACK = (100, 100, 100) # Gray (temporary occlusion)
COLOR_TRAIL = (255, 180, 50)          # Light cyan/blue for movement trail
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_OVERLAY_BG = (24, 24, 27)


@dataclass
class TrackedStudent:
    """Represents a student being tracked across video frames."""
    track_id: int
    box: List[int]                     # [x1, y1, x2, y2]
    center: Tuple[int, int]
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    consecutive_unobserved_frames: int = 0
    total_observed_frames: int = 1
    is_occluded: bool = False
    history: List[Tuple[int, int]] = field(default_factory=list)
    
    # Biometric identity placeholders (maintained separately with ArcFace verification)
    identity_id: Optional[str] = None
    identity_confidence: float = 0.0

    @property
    def observation_rate(self) -> float:
        """Ratio of frames in which the student was actively detected."""
        total_lifespan = (self.last_seen_frame - self.first_seen_frame + 1)
        return self.total_observed_frames / max(1, total_lifespan)


class StudentTracker:
    """
    ByteTrack-powered multi-student tracker for classroom surveillance.
    Operates on top of YOLO11 detections with temporary tracking IDs.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_occlusion_frames: int = 30,
        device: str = "cpu"
    ):
        """
        Initialize StudentTracker.

        Args:
            model_path: Path to YOLO weights (.pt)
            conf_threshold: Minimum confidence threshold
            iou_threshold: NMS IoU threshold
            max_occlusion_frames: Maximum frames to preserve lost tracks before pruning
            device: 'cpu' or 'cuda'
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_occlusion_frames = max_occlusion_frames
        self.device = device

        self.model_path = self._resolve_model_path(model_path)
        logger.info(f"Loading YOLO tracking model from: {self.model_path}")
        self.model = YOLO(self.model_path)

        # Active student tracks: track_id -> TrackedStudent
        self.tracks: Dict[int, TrackedStudent] = {}
        self.frame_index = 0

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        if model_path and os.path.exists(model_path):
            return str(Path(model_path).resolve())
        project_root = Path(__file__).resolve().parent.parent.parent
        standard_path = project_root / "models" / "yolo" / "yolo11n.pt"
        if standard_path.exists():
            return str(standard_path)
        return "yolo11n.pt"

    def reset(self):
        """Reset all tracking states."""
        self.tracks.clear()
        self.frame_index = 0
        logger.info("StudentTracker state reset.")

    def update(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame through YOLO11 with ByteTrack.

        Args:
            frame: Input BGR image (np.ndarray)

        Returns:
            Dictionary of active tracks, lost tracks, and frame telemetry.
        """
        start_time = time.perf_counter()
        self.frame_index += 1
        height, width = frame.shape[:2]

        # Run ByteTrack tracking (filtering person class = 0)
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],
            tracker="bytetrack.yaml",
            persist=True,
            device=self.device,
            verbose=False
        )

        detected_track_ids = set()
        active_this_frame: List[TrackedStudent] = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # ByteTrack assigns tracking ID when available
                if box.id is None:
                    continue

                track_id = int(box.id[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                xyxy = [int(v) for v in box.xyxy[0].cpu().numpy()]
                x1, y1 = max(0, xyxy[0]), max(0, xyxy[1])
                x2, y2 = min(width - 1, xyxy[2]), min(height - 1, xyxy[3])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                detected_track_ids.add(track_id)

                if track_id in self.tracks:
                    student = self.tracks[track_id]
                    student.box = [x1, y1, x2, y2]
                    student.center = center
                    student.confidence = round(conf, 3)
                    student.last_seen_frame = self.frame_index
                    student.consecutive_unobserved_frames = 0
                    student.total_observed_frames += 1
                    student.is_occluded = False
                    student.history.append(center)
                    # Limit history trail length
                    if len(student.history) > 30:
                        student.history.pop(0)
                else:
                    # New track initialized
                    student = TrackedStudent(
                        track_id=track_id,
                        box=[x1, y1, x2, y2],
                        center=center,
                        confidence=round(conf, 3),
                        first_seen_frame=self.frame_index,
                        last_seen_frame=self.frame_index,
                        history=[center]
                    )
                    self.tracks[track_id] = student

                active_this_frame.append(student)

        # Handle unobserved / occluded tracks
        pruned_tracks = []
        for track_id, student in list(self.tracks.items()):
            if track_id not in detected_track_ids:
                student.consecutive_unobserved_frames += 1
                student.is_occluded = True
                if student.consecutive_unobserved_frames > self.max_occlusion_frames:
                    pruned_tracks.append(track_id)
                    del self.tracks[track_id]

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "frame_index": self.frame_index,
            "active_tracks": [t for t in self.tracks.values() if not t.is_occluded],
            "occluded_tracks": [t for t in self.tracks.values() if t.is_occluded],
            "total_active_students": len(active_this_frame),
            "total_lifetime_tracks": len(self.tracks),
            "inference_time_ms": round(inference_time_ms, 2)
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracking_result: Dict[str, Any],
        fps: float = 0.0
    ) -> np.ndarray:
        """Draw tracked bounding boxes, temporary IDs, trajectory trails, and telemetry."""
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        # Draw active tracks
        for student in tracking_result["active_tracks"]:
            x1, y1, x2, y2 = student.box
            track_id = student.track_id
            conf = student.confidence

            # Draw trajectory path trail
            if len(student.history) > 1:
                for i in range(1, len(student.history)):
                    cv2.line(annotated, student.history[i - 1], student.history[i], COLOR_TRAIL, 2, cv2.LINE_AA)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_ACTIVE_TRACK, 2, cv2.LINE_AA)

            # Badge with temporary track ID (emphasizing temporary nature: 'Track #ID')
            label = f"Track #{track_id} ({conf:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 6)), (x1 + text_w + 6, y1), COLOR_ACTIVE_TRACK, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_TEXT_WHITE,
                1,
                cv2.LINE_AA
            )

        # Top telemetry bar
        bar_height = 36
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_height), COLOR_OVERLAY_BG, -1)
        cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

        active_count = len(tracking_result["active_tracks"])
        occluded_count = len(tracking_result["occluded_tracks"])
        telemetry = (
            f"ClassroomAI Tracker | Active: {active_count} | Occluded/Gaps: {occluded_count} | "
            f"Latency: {tracking_result['inference_time_ms']:.1f}ms"
        )
        if fps > 0:
            telemetry += f" | FPS: {fps:.1f}"

        cv2.putText(
            annotated,
            telemetry,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (100, 220, 100),
            1,
            cv2.LINE_AA
        )

        return annotated
