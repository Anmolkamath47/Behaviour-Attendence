"""
ClassroomAI - Complete End-to-End Classroom Observation & Attendance Pipeline
Synchronizes:
1. YOLO11 Student & Object Detection
2. ByteTrack Temporal Tracking
3. ArcFace 512-D Biometric Verification (with Informed Consent enforcement)
4. 6DRepNet & MediaPipe Posture & Head Orientation Analysis
5. TSM & XGBoost Multi-Modal Behaviour Classification (8 Classes)
6. 75% Rule Verified Physical Attendance Engine
7. Continuous Observation Timeline Generation
"""

import os
import sys
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.student_detection import StudentDetector
from src.tracking.student_tracker import StudentTracker
from src.face_recognition.face_detector import FaceDetector
from src.face_recognition.arcface import ArcFaceEmbedder
from src.face_recognition.matcher import FaceMatcher
from src.pose.head_pose import HeadPoseEstimator
from src.pose.mediapipe_analyzer import PostureAnalyzer
from src.behaviour.behaviour_engine import BehaviourEngine
from src.attendance.attendance_engine import AttendanceEngine, AttendanceStatus
from src.behaviour.timeline_analytics import StudentObservationTimeline

logger = logging.getLogger("ClassroomAI.Pipeline")


class ClassroomAIPipeline:
    """
    Central orchestration engine for live or offline classroom video observation.
    Designed to process 60-70 students from a rear-mounted classroom camera.
    """

    def __init__(
        self,
        session_id: str = "SESSION_001",
        lecture_duration_minutes: float = 60.0,
        yolo_model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        device: str = "cpu"
    ):
        self.session_id = session_id
        self.lecture_duration_minutes = lecture_duration_minutes

        logger.info("Initializing ClassroomAI Core ML Modules...")
        self.detector = StudentDetector(model_path=yolo_model_path, conf_threshold=conf_threshold, device=device)
        self.tracker = StudentTracker(model_path=yolo_model_path, conf_threshold=conf_threshold, device=device)
        self.face_detector = FaceDetector()
        self.embedder = ArcFaceEmbedder()
        self.face_matcher = FaceMatcher(similarity_threshold=0.55)
        self.behaviour_engine = BehaviourEngine()
        self.attendance_engine = AttendanceEngine(session_id, lecture_duration_minutes)

        # Track ID to Student ID cautious association mapping
        self.track_to_student: Dict[int, str] = {}
        # Timelines per verified student
        self.timelines: Dict[str, StudentObservationTimeline] = {}

        logger.info("ClassroomAI Pipeline fully initialized and ready.")

    def enroll_consented_student(
        self,
        student_id: str,
        roll_number: str,
        name: str,
        reference_face_rgb: Optional[np.ndarray] = None,
        consent_given: bool = True
    ):
        """Enrolls a student in the biometric registry with consent."""
        profile = self.face_matcher.enroll_student(
            student_id=student_id,
            roll_number=roll_number,
            full_name=name,
            consent_given=consent_given
        )
        if consent_given and reference_face_rgb is not None:
            emb = self.embedder.extract_embedding(reference_face_rgb)
            profile.add_reference_embedding(emb)

        self.attendance_engine.register_student(student_id, roll_number, name)
        self.timelines[student_id] = StudentObservationTimeline(student_id, roll_number, self.session_id)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int
    ) -> Dict[str, Any]:
        """
        Processes a single frame through the complete 7-stage ML pipeline.

        Returns:
            Structured telemetry and active observations.
        """
        start_time = time.perf_counter()

        # 1. ByteTrack Student Tracking
        tracking_result = self.tracker.update(frame)
        active_tracks = tracking_result["active_tracks"]

        # 2. Contextual Object Detections (Phones, Notebooks)
        detection_result = self.detector.detect(frame)
        phone_boxes = [obj["box"] for obj in detection_result["objects"] if obj["class_name"] == "cell phone"]

        frame_observations = []

        # 3. Process each tracked student
        for student in active_tracks:
            x1, y1, x2, y2 = student.box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Check if phone is nearby this student
            phone_nearby = any(
                abs(student.center[0] - ((px1 + px2) // 2)) < 60 and
                abs(student.center[1] - ((py1 + py2) // 2)) < 80
                for px1, py1, px2, py2 in phone_boxes
            )

            # 4. Face Recognition (ArcFace)
            matched_student_id = self.track_to_student.get(student.track_id)
            match_status = "UNKNOWN"
            is_verified = False

            if not matched_student_id:
                # Attempt cautious face verification
                face_res = self.face_detector.detect_face_in_crop(crop)
                if face_res is not None:
                    emb = self.embedder.extract_embedding(face_res["face_crop"])
                    match = self.face_matcher.match(emb)
                    if match.matched and match.student_id:
                        matched_student_id = match.student_id
                        self.track_to_student[student.track_id] = matched_student_id
                        match_status = "VERIFIED"
                        is_verified = True
            else:
                is_verified = True
                match_status = "VERIFIED"

            # 5. Behaviour Classification (TSM + XGBoost)
            obs = self.behaviour_engine.update_student(
                track_id=student.track_id,
                student_crop=crop,
                phone_nearby=phone_nearby,
                delta_time_sec=0.033
            )

            # 6. Physical Attendance Logging (Strict 75% Rule)
            if matched_student_id:
                self.attendance_engine.record_presence(
                    student_id=matched_student_id,
                    timestamp=timestamp,
                    is_verified_identity=is_verified,
                    confidence=student.confidence
                )
                if matched_student_id in self.timelines:
                    self.timelines[matched_student_id].record_activity_sample(
                        timestamp=timestamp,
                        activity=obs.primary_behaviour,
                        confidence=obs.confidence,
                        is_observable=True
                    )

            frame_observations.append({
                "track_id": student.track_id,
                "student_id": matched_student_id,
                "box": student.box,
                "behaviour": obs.primary_behaviour,
                "confidence": obs.confidence,
                "is_verified": is_verified
            })

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "latency_ms": round(latency_ms, 2),
            "active_students": len(active_tracks),
            "observations": frame_observations
        }

    def benchmark_scaling(
        self,
        student_counts: List[int] = [5, 15, 30, 65],
        frames_per_benchmark: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmarks inference latency across 5, 15, 30, and 60-70 students.
        """
        logger.info("Executing ClassroomAI Scaling Benchmark...")
        benchmark_results = {}

        # 720p base frame
        base_frame = np.full((720, 1280, 3), 200, dtype=np.uint8)

        for count in student_counts:
            # Synthesize grid of student bodies
            test_frame = base_frame.copy()
            cols = 10
            rows = int(np.ceil(count / cols))
            box_w, box_h = 70, 110

            for i in range(count):
                r = i // cols
                c = i % cols
                bx1 = 50 + c * 115
                by1 = 80 + r * 140
                bx2 = min(1270, bx1 + box_w)
                by2 = min(710, by1 + box_h)
                cv2.rectangle(test_frame, (bx1, by1), (bx2, by2), (120, 100, 80), -1)
                cv2.circle(test_frame, ((bx1 + bx2) // 2, by1 + 25), 18, (180, 150, 130), -1)

            latencies = []
            for f in range(frames_per_benchmark):
                res = self.process_frame(test_frame, timestamp=float(f), frame_idx=f)
                latencies.append(res["latency_ms"])

            avg_latency = float(np.mean(latencies))
            fps = round(1000.0 / avg_latency, 1) if avg_latency > 0 else 0.0

            benchmark_results[f"{count}_students"] = {
                "student_count": count,
                "avg_latency_ms": round(avg_latency, 2),
                "estimated_cpu_fps": fps,
                "cpu_feasibility": "REAL_TIME" if fps >= 15 else "OFFLINE_BATCH"
            }
            logger.info(f"Benchmark [{count} students]: {avg_latency:.2f} ms ({fps} FPS on CPU)")

        return benchmark_results
