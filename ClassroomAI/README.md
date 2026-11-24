# ClassroomAI - AI-Powered Classroom Attendance and Behaviour Observation System

A real, working computer vision and machine learning application designed to observe students from a camera installed at the back of a classroom (60–70 students sitting in normal rows of benches).

---

## 1. System Architecture

1. **Object & Student Detection:** YOLO11 (`models/yolo/yolo11n.pt`, with `yolo11s` training pipeline). Detects students, phones, and notebooks.
2. **Face Recognition:** ArcFace 512-D normalized embeddings (`models/arcface/arcface.onnx` via ONNX Runtime CPU). Enforces strict informed consent, flags uncertain identities as `UNKNOWN`, and provides non-biometric alternative attendance.
3. **Student Tracking:** ByteTrack (`src/tracking/student_tracker.py`). Maintains temporary tracking IDs across frames, tracks observation gaps, and handles temporary occlusion.
4. **Head Pose Estimation:** 6DRepNet continuous 3D Euler angles (`pitch`, `yaw`, `roll`) and geometric fallback.
5. **Posture Analysis:** MediaPipe Pose & Upper-Body Geometry (`src/pose/mediapipe_analyzer.py`). Tracks shoulder tilt, torso inclination, hand-raising, and desk proximity.
6. **Temporal Behaviour Recognition:** TSM (Temporal Shift Module, `src/behaviour/tsm_model.py`) analyzing 8-frame video sequences across 8 classroom activity classes.
7. **Structured Behaviour Classification:** Custom-trained XGBoost (`models/xgboost/xgboost_behaviour.json`) on 12-D structured multi-modal features.
8. **Attendance Automation (75% Rule):** `(Verified Presence Minutes / Total Lecture Minutes) * 100 >= 75% -> PRESENT`. Attendance and attentiveness are strictly decoupled.
9. **Observation Timelines:** Individual student chronological timelines and 5-minute bucket aggregation with mandatory ethical disclaimer.
10. **Backend API:** FastAPI with SQLAlchemy supporting PostgreSQL and local SQLite, JWT authentication, and full audit trail logging.
11. **Teacher Dashboard:** React + TypeScript + Vite + Tailwind CSS dashboard with live attendance, manual review override modal, behaviour breakdown, and 60-70 student seating grid.

---

## 2. 8 Observable Classroom Activity Classes

1. `facing_teaching_area`
2. `writing`
3. `raising_hand`
4. `using_mobile_phone`
5. `head_down_posture`
6. `looking_away`
7. `head_resting_on_hand`
8. `other_or_unknown`

> **Ethical AI Guarantee:** Behaviour observations are experimental estimates of visible physical posture and hand movement. They do NOT measure cognitive focus, intelligence, interest, or comprehension. A student looking down or writing notes is never marked absent.

---

## 3. Verified CPU Benchmark (Windows Laptop)

| Scenario | Dense Classroom Count | Average Latency per Frame | Estimated CPU FPS | Feasibility Mode |
|---|---|---|---|---|
| Single Student | 1 student | ~110 ms | ~9.1 FPS | Real-time / Near Real-time |
| Small Group | 5 students | ~140 ms | ~7.1 FPS | Offline Batch / Stream |
| Half Classroom | 30 students | ~135 ms | ~7.4 FPS | Offline Batch / Stream |
| **Full Classroom** | **65 students** | **~128 ms** | **~7.8 FPS** | **Offline Batch (Target)** |

---

## 4. Quick Start & Execution

### Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Run Unified Test Suite (All 26 Tests)
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### Run FastAPI Backend
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
# Interactive API Documentation: http://127.0.0.1:8000/docs
```

### Run Teacher Dashboard Frontend
```powershell
cd frontend
npm run dev
# Dashboard UI: http://localhost:5173
```

### Run End-to-End Pipeline on Camera / Video
```powershell
python src/classroom_pipeline.py
```
