"""
ClassroomAI - Pydantic Request & Response Schemas
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# Authentication
class UserRegister(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "teacher"


class UserLogin(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    full_name: str
    role: str


# Students
class StudentCreate(BaseModel):
    roll_number: str
    full_name: str
    email: Optional[str] = None
    consent_given: bool = False
    alternative_attendance_mode: str = "MANUAL_REVIEW"


class StudentResponse(BaseModel):
    id: str
    roll_number: str
    full_name: str
    email: Optional[str]
    consent_given: bool
    alternative_attendance_mode: str
    created_at: datetime

    class Config:
        from_attributes = True


class ConsentUpdate(BaseModel):
    consent_given: bool
    alternative_mode: Optional[str] = "MANUAL_REVIEW"


# Classroom & Seating
class ClassroomCreate(BaseModel):
    name: str
    room_number: str
    camera_source: str = "0"
    total_capacity: int = 70


class SeatingAssignmentCreate(BaseModel):
    student_id: str
    row_number: int
    bench_number: int
    position: str = "left"


# Lecture Session
class SessionStart(BaseModel):
    classroom_id: str
    subject_title: str
    duration_minutes: float = 60.0


class SessionResponse(BaseModel):
    id: str
    classroom_id: str
    subject_title: str
    duration_minutes: float
    status: str
    start_time: datetime

    class Config:
        from_attributes = True


# Attendance & Manual Review
class ManualAttendanceOverride(BaseModel):
    student_id: str
    new_status: str  # "PRESENT", "BELOW_REQUIRED_ATTENDANCE"
    reason: str


class AttendanceResponse(BaseModel):
    id: str
    session_id: str
    student_id: str
    roll_number: str
    student_name: str
    total_lecture_minutes: float
    verified_presence_minutes: float
    presence_percentage: float
    status: str
    arrival_time: Optional[str]
    departure_time: Optional[str]
    manual_review_required: bool
    manual_override: bool
    audit_notes: Optional[str]


# Behaviour Analytics
class BehaviourTimelineResponse(BaseModel):
    student_id: str
    roll_number: str
    student_name: str
    breakdown_minutes: Dict[str, float]
    timeline_buckets: List[Dict[str, Any]]
    disclaimer: str
