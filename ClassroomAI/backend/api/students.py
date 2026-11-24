"""
ClassroomAI - Student Management & Consent Endpoints
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database.connection import get_db
from ..database.models import Student, ConsentRecord, SeatingAssignment
from ..schemas.schemas import StudentCreate, StudentResponse, ConsentUpdate, SeatingAssignmentCreate

router = APIRouter(prefix="/students", tags=["Students"])


@router.get("", response_model=List[StudentResponse])
def list_students(db: Session = Depends(get_db)):
    students = db.query(Student).all()
    results = []
    for s in students:
        consent = s.consent
        results.append(
            StudentResponse(
                id=s.id,
                roll_number=s.roll_number,
                full_name=s.full_name,
                email=s.email,
                consent_given=consent.consent_given if consent else False,
                alternative_attendance_mode=consent.alternative_attendance_mode if consent else "MANUAL_REVIEW",
                created_at=s.created_at
            )
        )
    return results


@router.post("", response_model=StudentResponse)
def create_student(s_in: StudentCreate, db: Session = Depends(get_db)):
    existing = db.query(Student).filter(Student.roll_number == s_in.roll_number).first()
    if existing:
        raise HTTPException(status_code=400, detail="Student with this roll number already exists.")

    student = Student(
        roll_number=s_in.roll_number,
        full_name=s_in.full_name,
        email=s_in.email
    )
    db.add(student)
    db.flush()

    consent = ConsentRecord(
        student_id=student.id,
        consent_given=s_in.consent_given,
        alternative_attendance_mode=s_in.alternative_attendance_mode
    )
    db.add(consent)
    db.commit()
    db.refresh(student)

    return StudentResponse(
        id=student.id,
        roll_number=student.roll_number,
        full_name=student.full_name,
        email=student.email,
        consent_given=s_in.consent_given,
        alternative_attendance_mode=s_in.alternative_attendance_mode,
        created_at=student.created_at
    )


@router.put("/{student_id}/consent")
def update_consent(student_id: str, consent_in: ConsentUpdate, db: Session = Depends(get_db)):
    consent = db.query(ConsentRecord).filter(ConsentRecord.student_id == student_id).first()
    if not consent:
        raise HTTPException(status_code=404, detail="Student consent record not found.")

    consent.consent_given = consent_in.consent_given
    if consent_in.alternative_mode:
        consent.alternative_attendance_mode = consent_in.alternative_mode
    db.commit()
    return {"message": "Consent updated successfully", "consent_given": consent.consent_given}


@router.post("/seating")
def assign_seating(seat_in: SeatingAssignmentCreate, db: Session = Depends(get_db)):
    seat = SeatingAssignment(
        classroom_id=seat_in.classroom_id,
        student_id=seat_in.student_id,
        row_number=seat_in.row_number,
        bench_number=seat_in.bench_number,
        position=seat_in.position
    )
    db.add(seat)
    db.commit()
    return {"message": "Seating assigned successfully"}
