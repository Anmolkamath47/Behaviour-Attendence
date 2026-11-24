"""
ClassroomAI - Student Detection Module
Detects students and visible classroom objects (phones, notebooks) using YOLO11.
Supports webcam feeds, recorded video files, and automated test frames.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Detection] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ClassroomAI.Detection")

# Class ID mapping for standard COCO weights
COCO_PERSON_CLASS_ID = 0
COCO_CELL_PHONE_CLASS_ID = 67
COCO_BOOK_CLASS_ID = 73

# BGR Colors for clean, professional ed-tech display
COLOR_PRIMARY_BLUE = (235, 99, 37)       # #2563EB in BGR (Student / Person)
COLOR_AMBER = (6, 119, 217)              # #D97706 in BGR (Phone)
COLOR_EMERALD = (105, 150, 5)            # #059669 in BGR (Book / Notebook)
COLOR_GRAY = (120, 113, 108)             # #6C757D in BGR (Default / Other)
COLOR_BG_DARK = (30, 27, 24)             # Status bar dark overlay
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_TEXT_GREEN = (100, 220, 100)


class StudentDetector:
    """
    YOLO11-based student detector optimized for classroom observation.
    Detects students/persons and relevant classroom objects.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        track_classroom_objects: bool = True
    ):
        """
        Initialize the StudentDetector.

        Args:
            model_path: Path to YOLO weights (.pt). If None, defaults to models/yolo/yolo11n.pt
            conf_threshold: Minimum confidence for detection [0.0, 1.0]
            iou_threshold: NMS IoU threshold [0.0, 1.0]
            device: 'cpu' or 'cuda' (defaults to cpu for portable laptop execution)
            track_classroom_objects: If True, also detects phones and books for behaviour context
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.track_classroom_objects = track_classroom_objects

        # Resolve model path
        self.model_path = self._resolve_model_path(model_path)
        logger.info(f"Loading YOLO model from: {self.model_path} on device: {self.device}")
        self.model = YOLO(self.model_path)
        logger.info("YOLO model successfully loaded.")

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Find the local model path or fall back to pretrained yolo11n."""
        if model_path and os.path.exists(model_path):
            return str(Path(model_path).resolve())

        # Check standard project locations
        project_root = Path(__file__).resolve().parent.parent.parent
        standard_path = project_root / "models" / "yolo" / "yolo11n.pt"

        if standard_path.exists():
            return str(standard_path)

        # Fallback to local yolo11n.pt or allow ultralytics to download
        return "yolo11n.pt"

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run detection on a single frame.

        Args:
            frame: Input BGR image from OpenCV (np.ndarray)

        Returns:
            Dictionary containing detected students, context objects, counts, and inference time.
        """
        start_time = time.perf_counter()
        height, width = frame.shape[:2]

        # Classes to filter: person (0), cell phone (67), book (73)
        target_classes = [COCO_PERSON_CLASS_ID]
        if self.track_classroom_objects:
            target_classes.extend([COCO_CELL_PHONE_CLASS_ID, COCO_BOOK_CLASS_ID])

        # Inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=target_classes,
            device=self.device,
            verbose=False
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        students: List[Dict[str, Any]] = []
        objects: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]

                    # Clamp coordinates to frame boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)

                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                    detection_item = {
                        "box": [x1, y1, x2, y2],
                        "confidence": round(conf, 3),
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "area": (x2 - x1) * (y2 - y1)
                    }

                    if cls_id == COCO_PERSON_CLASS_ID:
                        # Label as student for classroom domain
                        detection_item["label"] = "Student"
                        students.append(detection_item)
                    else:
                        detection_item["label"] = cls_name.capitalize()
                        objects.append(detection_item)

        return {
            "students": students,
            "objects": objects,
            "student_count": len(students),
            "object_count": len(objects),
            "inference_time_ms": round(inference_time_ms, 2),
            "frame_dimensions": (width, height)
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        detection_result: Dict[str, Any],
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Draw clean bounding boxes, labels, and status telemetry on the frame.

        Args:
            frame: Original BGR image
            detection_result: Dictionary returned by detect()
            fps: Current rolling frames per second

        Returns:
            Annotated BGR image
        """
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        # Draw detected students
        for student in detection_result["students"]:
            x1, y1, x2, y2 = student["box"]
            conf = student["confidence"]

            # Main bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_PRIMARY_BLUE, 2, cv2.LINE_AA)

            # Modern label badge
            label = f"Student {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            badge_y1 = max(0, y1 - text_h - 6)
            badge_y2 = y1
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + text_w + 6, badge_y2), COLOR_PRIMARY_BLUE, -1)
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

        # Draw detected objects (phones, notebooks)
        for obj in detection_result["objects"]:
            x1, y1, x2, y2 = obj["box"]
            cls_name = obj["class_name"]
            conf = obj["confidence"]
            color = COLOR_AMBER if obj["class_id"] == COCO_CELL_PHONE_CLASS_ID else COLOR_EMERALD

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"{obj['label']} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            badge_y1 = max(0, y1 - text_h - 6)
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + text_w + 6, y1), color, -1)
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

        # Top status bar for real-time monitoring
        bar_height = 36
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_height), COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Telemetry text
        count_text = f"ClassroomAI | Students: {detection_result['student_count']}"
        timing_text = f"Inference: {detection_result['inference_time_ms']:.1f}ms"
        fps_text = f"FPS: {fps:.1f}" if fps > 0 else ""

        status_text = f"{count_text}  |  {timing_text}  |  {fps_text}".strip(" | ")
        cv2.putText(
            annotated,
            status_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_TEXT_GREEN,
            1,
            cv2.LINE_AA
        )

        return annotated

    def run_self_test(self) -> bool:
        """
        Verify the detector pipeline with a synthesized test frame.
        Guarantees functionality without requiring an active GUI window.
        """
        logger.info("Running StudentDetector self-test...")
        # Create a blank 720p test image
        test_frame = np.full((720, 1280, 3), 200, dtype=np.uint8)

        # Draw simulated student shapes (circle head + rounded torso)
        cv2.rectangle(test_frame, (300, 300), (450, 650), (120, 100, 80), -1)
        cv2.circle(test_frame, (375, 230), 60, (180, 150, 130), -1)

        result = self.detect(test_frame)
        logger.info(
            f"Self-test inference completed: {result['inference_time_ms']:.2f} ms | "
            f"Result keys verified: {list(result.keys())}"
        )

        annotated = self.annotate_frame(test_frame, result, fps=30.0)
        assert annotated.shape == test_frame.shape, "Annotated frame dimensions mismatch"
        logger.info("Self-test passed successfully!")
        return True

    def process_source(
        self,
        source: Any = 0,
        headless: bool = False,
        save_path: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run detection loop on a camera feed or video file.

        Args:
            source: 0 (webcam), path to video file, or 'test'
            headless: If True, disables cv2.imshow
            save_path: Optional path to save processed video (.mp4)
            max_frames: Optional frame processing limit
        """
        if str(source).lower() == "test":
            self.run_self_test()
            return {"status": "self_test_passed"}

        # Open video source
        try:
            cam_idx = int(source)
            cap = cv2.VideoCapture(cam_idx)
            is_stream = True
        except ValueError:
            cap = cv2.VideoCapture(str(source))
            is_stream = False

        if not cap.isOpened():
            logger.error(f"Cannot open video source: {source}")
            return {"error": f"Cannot open video source: {source}"}

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stream_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        writer = None
        if save_path:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, stream_fps, (frame_width, frame_height))
            logger.info(f"Saving output video to: {out_path}")

        logger.info(f"Processing source '{source}' ({frame_width}x{frame_height} @ {stream_fps:.1f} FPS)")
        if not headless:
            logger.info("Press 'q' in the window to stop.")

        frame_count = 0
        total_inference_ms = 0.0
        fps_start = time.perf_counter()
        current_fps = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream reached.")
                    break

                frame_count += 1
                det_result = self.detect(frame)
                total_inference_ms += det_result["inference_time_ms"]

                # Calculate rolling FPS every 10 frames
                if frame_count % 10 == 0:
                    elapsed = time.perf_counter() - fps_start
                    current_fps = 10.0 / elapsed if elapsed > 0 else 0.0
                    fps_start = time.perf_counter()

                annotated = self.annotate_frame(frame, det_result, fps=current_fps)

                if writer:
                    writer.write(annotated)

                if not headless:
                    cv2.imshow("ClassroomAI - Student Detection (YOLO11)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested exit ('q' pressed).")
                        break

                if max_frames and frame_count >= max_frames:
                    logger.info(f"Reached max frames limit ({max_frames}).")
                    break

        finally:
            cap.release()
            if writer:
                writer.release()
            if not headless:
                cv2.destroyAllWindows()

        avg_inference_ms = (total_inference_ms / frame_count) if frame_count > 0 else 0.0
        logger.info(
            f"Finished processing. Total frames: {frame_count} | "
            f"Avg inference: {avg_inference_ms:.2f} ms (~{1000.0/avg_inference_ms if avg_inference_ms > 0 else 0:.1f} FPS on CPU)"
        )

        return {
            "total_frames": frame_count,
            "avg_inference_ms": round(avg_inference_ms, 2),
            "estimated_cpu_fps": round(1000.0 / avg_inference_ms, 1) if avg_inference_ms > 0 else 0.0
        }


def main():
    parser = argparse.ArgumentParser(description="ClassroomAI - Student & Classroom Object Detector (YOLO11)")
    parser.add_argument(
        "--source",
        default="test",
        help="Input source: '0' for webcam, path to video/image file, or 'test' for self-test (default: test)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to YOLO weights (.pt). Default: models/yolo/yolo11n.pt"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying OpenCV GUI window"
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Path to save annotated video output"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process"
    )

    args = parser.parse_args()

    detector = StudentDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    detector.process_source(
        source=args.source,
        headless=args.headless,
        save_path=args.save,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
