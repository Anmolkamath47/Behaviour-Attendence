"""
ClassroomAI - SQLAlchemy Database Models (12 Tables)
Strictly adheres to Database Design Section 9.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, Index
)
from sqlalchemy.orm import relationship

from .connection import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


# 1. Users Table (Teachers & Admins)
class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(String(20), default="teacher", nullable=False)  # "teacher" or "admin"
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("LectureSession", back_populates="teacher")


# 2. Students Table
class Student(Base):
    __tablename__ = "students"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    roll_number = Column(String(50), unique=True, nullable=False, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    consent = relationship("ConsentRecord", uselist=False, back_populates="student")
    seating = relationship("SeatingAssignment", back_populates="student")
    attendance_records = relationship("AttendanceRecord", back_populates="student")


# 3. Classrooms Table
class Classroom(Base):
    __tablename__ = "classrooms"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    room_number = Column(String(50), nullable=False)
    camera_source = Column(String(255), default="0")  # Camera index or RTSP / video file path
    total_capacity = Column(Integer, default=70)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("LectureSession", back_populates="classroom")
    seating = relationship("SeatingAssignment", back_populates="classroom")


# 4. Classroom-Students Association
class ClassroomStudent(Base):
    __tablename__ = "classroom_students"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    classroom_id = Column(String(36), ForeignKey("classrooms.id", ondelete="CASCADE"), nullable=False)
    student_id = Column(String(36), ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    enrolled_at = Column(DateTime, default=datetime.utcnow)


# 5. Seating Assignments Table (Row & Bench positions for 60-70 students)
class SeatingAssignment(Base):
    __tablename__ = "seating_assignments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    classroom_id = Column(String(36), ForeignKey("classrooms.id", ondelete="CASCADE"), nullable=False)
    student_id = Column(String(36), ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    row_number = Column(Integer, nullable=False)
    bench_number = Column(Integer, nullable=False)
    position = Column(String(20), default="left")  # "left", "center", "right"

    classroom = relationship("Classroom", back_populates="seating")
    student = relationship("Student", back_populates="seating")


# 6. Consent Records Table
class ConsentRecord(Base):
    __tablename__ = "consent_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    student_id = Column(String(36), ForeignKey("students.id", ondelete="CASCADE"), unique=True, nullable=False)
    consent_given = Column(Boolean, default=False, nullable=False)
    consent_timestamp = Column(DateTime, nullable=True)
    alternative_attendance_mode = Column(String(50), default="MANUAL_REVIEW")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    student = relationship("Student", back_populates="consent")


# 7. Lecture Sessions Table
class LectureSession(Base):
    __tablename__ = "lecture_sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    classroom_id = Column(String(36), ForeignKey("classrooms.id"), nullable=False)
    teacher_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    subject_title = Column(String(150), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    duration_minutes = Column(Float, default=60.0)
    status = Column(String(20), default="ACTIVE")  # "ACTIVE", "COMPLETED"

    classroom = relationship("Classroom", back_populates="sessions")
    teacher = relationship("User", back_populates="sessions")
    attendance = relationship("AttendanceRecord", back_populates="session")


# 8. Attendance Records Table (Strict 75% Rule Decisions)
class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("lecture_sessions.id", ondelete="CASCADE"), nullable=False)
    student_id = Column(String(36), ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    total_lecture_minutes = Column(Float, nullable=False)
    verified_presence_minutes = Column(Float, default=0.0)
    uncertain_minutes = Column(Float, default=0.0)
    presence_percentage = Column(Float, default=0.0)
    status = Column(String(30), default="BELOW_REQUIRED_ATTENDANCE")
    arrival_time = Column(String(20), nullable=True)
    departure_time = Column(String(20), nullable=True)
    manual_override = Column(Boolean, default=False)
    manual_review_required = Column(Boolean, default=False)
    audit_notes = Column(Text, nullable=True)

    session = relationship("LectureSession", back_populates="attendance")
    student = relationship("Student", back_populates="attendance_records")


# 9. Presence Observations Table (Time-Binned)
class PresenceObservation(Base):
    __tablename__ = "presence_observations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("lecture_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    student_id = Column(String(36), ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(Float, nullable=False)
    time_bin_second = Column(Integer, nullable=False)
    is_verified = Column(Boolean, default=False)
    confidence = Column(Float, default=0.0)


# 10. Behaviour Observations Table (8 Activity Classes)
class BehaviourObservation(Base):
    __tablename__ = "behaviour_observations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("lecture_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    student_id = Column(String(36), ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(Float, nullable=False)
    primary_behaviour = Column(String(50), nullable=False)
    confidence = Column(Float, default=0.0)
    tsm_prediction = Column(String(50), nullable=True)
    xgboost_prediction = Column(String(50), nullable=True)
    is_observable = Column(Boolean, default=True)


# 11. Model Versions Table
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_name = Column(String(100), nullable=False)  # "YOLO11s", "ArcFace", "TSM", "XGBoost"
    version = Column(String(50), nullable=False)
    weights_path = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    deployed_at = Column(DateTime, default=datetime.utcnow)


# 12. Audit Logs Table (Full Traceability for Teacher Decisions)
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("lecture_sessions.id", ondelete="CASCADE"), nullable=False)
    student_id = Column(String(36), ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    previous_status = Column(String(50), nullable=False)
    new_status = Column(String(50), nullable=False)
    modified_by = Column(String(100), nullable=False)  # Teacher ID / Name
    reason = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
