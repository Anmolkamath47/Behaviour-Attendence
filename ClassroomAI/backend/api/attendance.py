"""
ClassroomAI - Attendance & Manual Review Endpoints
"""

import io
import csv
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from ..database.connection import get_db
from ..database.models import AttendanceRecord, LectureSession, AuditLog, Student
from ..schemas.schemas import AttendanceResponse, ManualAttendanceOverride

router = APIRouter(prefix="/attendance", tags=["Attendance"])


@router.get("/session/{session_id}", response_model=List[AttendanceResponse])
def get_session_attendance(session_id: str, db: Session = Depends(get_db)):
    records = db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session_id).all()
    results = []
    for r in records:
        student = db.query(Student).filter(Student.id == r.student_id).first()
        results.append(
            AttendanceResponse(
                id=r.id,
                session_id=r.session_id,
                student_id=r.student_id,
                roll_number=student.roll_number if student else "N/A",
                student_name=student.full_name if student else "Unknown",
                total_lecture_minutes=r.total_lecture_minutes,
                verified_presence_minutes=r.verified_presence_minutes,
                presence_percentage=r.presence_percentage,
                status=r.status,
                arrival_time=r.arrival_time,
                departure_time=r.departure_time,
                manual_review_required=r.manual_review_required,
                manual_override=r.manual_override,
                audit_notes=r.audit_notes
            )
        )
    return results


@router.post("/override")
def override_attendance(
    override_in: ManualAttendanceOverride,
    session_id: str,
    teacher_name: str = "Teacher",
    db: Session = Depends(get_db)
):
    record = db.query(AttendanceRecord).filter(
        AttendanceRecord.session_id == session_id,
        AttendanceRecord.student_id == override_in.student_id
    ).first()

    if not record:
        raise HTTPException(status_code=404, detail="Attendance record not found.")

    old_status = record.status
    record.status = override_in.new_status
    record.manual_override = True
    record.manual_review_required = False
    record.audit_notes = f"Overridden by {teacher_name}: {override_in.reason}"

    # Log to audit trail
    audit_entry = AuditLog(
        session_id=session_id,
        student_id=override_in.student_id,
        previous_status=old_status,
        new_status=override_in.new_status,
        modified_by=teacher_name,
        reason=override_in.reason
    )
    db.add(audit_entry)
    db.commit()

    return {"message": "Attendance successfully overridden", "audit_logged": True}


@router.get("/export/csv/{session_id}")
def export_csv(session_id: str, db: Session = Depends(get_db)):
    records = db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session_id).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Roll Number", "Student Name", "Total Minutes", "Verified Minutes",
        "Presence %", "Status", "Arrival", "Departure", "Manual Override", "Audit Notes"
    ])

    for r in records:
        student = db.query(Student).filter(Student.id == r.student_id).first()
        writer.writerow([
            student.roll_number if student else "N/A",
            student.full_name if student else "Unknown",
            r.total_lecture_minutes,
            r.verified_presence_minutes,
            r.presence_percentage,
            r.status,
            r.arrival_time or "N/A",
            r.departure_time or "N/A",
            "YES" if r.manual_override else "NO",
            r.audit_notes or ""
        ])

    csv_data = output.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_{session_id}.csv"}
    )
