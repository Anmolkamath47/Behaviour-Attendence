"""
ClassroomAI - FastAPI Application Server
Initializes database schema, mounts API routers, configures CORS, and exposes health telemetry.
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database.connection import engine, Base
from backend.api import auth, students, classrooms, attendance, analytics

# Create all database tables automatically on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="ClassroomAI Backend API",
    description="AI-powered Classroom Attendance & Behaviour Observation System",
    version="1.0.0"
)

# CORS configuration for local React Vite development and production deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(auth.router, prefix="/api")
app.include_router(students.router, prefix="/api")
app.include_router(classrooms.router, prefix="/api")
app.include_router(attendance.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")


@app.get("/")
def root():
    return {
        "system": "ClassroomAI Backend",
        "version": "1.0.0",
        "status": "HEALTHY",
        "docs_url": "/docs"
    }


@app.get("/health")
def health_check():
    return {"status": "UP", "database": "CONNECTED"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
