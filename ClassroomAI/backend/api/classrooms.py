"""
ClassroomAI - Classrooms & Lecture Sessions Endpoints
"""

from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database.connection import get_db
from ..database.models import Classroom, LectureSession, User
from ..schemas.schemas import ClassroomCreate, SessionStart, SessionResponse

router = APIRouter(prefix="/classrooms", tags=["Classrooms"])


@router.get("", response_model=List[dict])
def list_classrooms(db: Session = Depends(get_db)):
    classrooms = db.query(Classroom).all()
    return [
        {
            "id": c.id,
            "name": c.name,
            "room_number": c.room_number,
            "camera_source": c.camera_source,
            "total_capacity": c.total_capacity
        }
        for c in classrooms
    ]


@router.post("", response_model=dict)
def create_classroom(c_in: ClassroomCreate, db: Session = Depends(get_db)):
    classroom = Classroom(
        name=c_in.name,
        room_number=c_in.room_number,
        camera_source=c_in.camera_source,
        total_capacity=c_in.total_capacity
    )
    db.add(classroom)
    db.commit()
    db.refresh(classroom)
    return {"id": classroom.id, "name": classroom.name, "room_number": classroom.room_number}


@router.post("/sessions/start", response_model=SessionResponse)
def start_session(s_in: SessionStart, teacher_id: str = None, db: Session = Depends(get_db)):
    if not teacher_id:
        teacher = db.query(User).first()
        teacher_id = teacher.id if teacher else "DEFAULT_TEACHER"

    session = LectureSession(
        classroom_id=s_in.classroom_id,
        teacher_id=teacher_id,
        subject_title=s_in.subject_title,
        duration_minutes=s_in.duration_minutes,
        status="ACTIVE"
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.post("/sessions/{session_id}/stop")
def stop_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(LectureSession).filter(LectureSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    session.status = "COMPLETED"
    session.end_time = datetime.utcnow()
    db.commit()
    return {"message": "Session completed successfully", "session_id": session_id}
