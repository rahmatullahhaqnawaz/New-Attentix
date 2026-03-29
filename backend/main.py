"""
attentix — main.py
FastAPI server exposing all endpoints for the Lovable frontend.
Run with: uvicorn main:app --reload --port 8000
"""

import os
import time
import shutil
import base64
import tempfile
import threading
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from face_engine import FaceEngine
from video_processor import process_video, process_batch

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Attentix API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state ───────────────────────────────────────────────────────────
_live_engine  = FaceEngine()          # one engine per live session
_state        = {
    "speed":           1.0,
    "current_quiz":    None,           # dict or None
    "students":        [],             # list of processed student results
    "session_active":  False,
}
_state_lock = threading.Lock()


# ── Pydantic models ───────────────────────────────────────────────────────────
class FramePayload(BaseModel):
    frame_b64: str                     # base64-encoded JPEG from webcam
    timestamp: float = 0.0

class SpeedPayload(BaseModel):
    speed: float                       # 0.75 | 1.0 | 1.5 | 2.0

class QuizPayload(BaseModel):
    question: str
    options: list[str]                 # exactly 3
    correct_index: int                 # 0, 1, or 2
    target: str = "all"               # "all" | "low"

class QuizAnswerPayload(BaseModel):
    student_id: str
    answer_index: int


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "Attentix API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


# ─────────────────────────────────────────────────────────────────────────────
# LIVE ENGAGEMENT  (student webcam mode)
# POST /engagement  — send a webcam frame, get score back
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/engagement")
async def engagement(payload: FramePayload):
    """
    Receive a base64 JPEG frame from the student's webcam.
    Returns score, state, all 6 signals, events, action, teacher_alert.
    """
    try:
        img_bytes = base64.b64decode(payload.frame_b64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid frame: {e}")

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = _live_engine.analyse_frame(frame, payload.timestamp)

    # Auto-update speed state based on score
    with _state_lock:
        if result["score"] >= 70:
            _state["speed"] = 1.5
        elif result["score"] >= 40:
            _state["speed"] = 1.0
        else:
            _state["speed"] = 1.0     # paused by quiz

    result["current_speed"] = _state["speed"]
    result["quiz_pending"]  = _state["current_quiz"] is not None

    return result


# GET version — for polling (returns last known state, no frame needed)
@app.get("/engagement")
async def engagement_get():
    """Polling endpoint — returns current state without a new frame."""
    with _state_lock:
        return {
            "speed":        _state["speed"],
            "quiz_pending": _state["current_quiz"] is not None,
            "quiz":         _state["current_quiz"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO UPLOAD  (student upload mode + teacher batch)
# POST /upload  — single video
# POST /upload/batch  — up to 10 videos at once
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_single(
    file: UploadFile = File(...),
    student_name: str = Form(default="Student"),
):
    """
    Upload a single video file (MP4/MOV/AVI).
    Returns full engagement report with timeline and events.
    """
    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        report = process_video(tmp_path, student_name)
    finally:
        os.unlink(tmp_path)

    if "error" in report:
        raise HTTPException(status_code=422, detail=report["error"])

    # Store in students list
    with _state_lock:
        # Replace existing entry for same student or append
        existing = [i for i, s in enumerate(_state["students"]) if s["student"] == student_name]
        if existing:
            _state["students"][existing[0]] = report
        else:
            _state["students"].append(report)

    return report


@app.post("/upload/batch")
async def upload_batch(
    files: list[UploadFile] = File(...),
    student_names: str = Form(default=""),   # comma-separated names
):
    """
    Upload up to 10 video files at once.
    Returns class-wide report + per-student results.
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 videos per batch")

    names = [n.strip() for n in student_names.split(",")] if student_names else []
    video_paths = []
    tmp_paths   = []

    try:
        for i, f in enumerate(files):
            suffix = os.path.splitext(f.filename)[-1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(f.file, tmp)
                tmp_paths.append(tmp.name)
            name = names[i] if i < len(names) else f"Student {i + 1}"
            video_paths.append((tmp.name, name))

        batch_report = process_batch(video_paths)

    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass

    if "error" in batch_report:
        raise HTTPException(status_code=422, detail=batch_report["error"])

    # Store all students
    with _state_lock:
        _state["students"] = batch_report["students"]

    return batch_report


# ─────────────────────────────────────────────────────────────────────────────
# STUDENTS  (teacher dashboard roster)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/students")
def get_students():
    """Return all processed student results, sorted by score descending."""
    with _state_lock:
        sorted_students = sorted(
            _state["students"],
            key=lambda s: s.get("summary", {}).get("avg_score", 0),
            reverse=True,
        )
    return {"students": sorted_students, "count": len(sorted_students)}


@app.delete("/students")
def clear_students():
    """Reset the student roster (start a new session)."""
    with _state_lock:
        _state["students"] = []
        _live_engine.reset()
    return {"status": "cleared"}


# ─────────────────────────────────────────────────────────────────────────────
# SPEED CONTROL
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/speed")
def set_speed(payload: SpeedPayload):
    """Teacher sets playback speed. Valid: 0.75, 1.0, 1.5, 2.0"""
    allowed = {0.75, 1.0, 1.5, 2.0}
    if payload.speed not in allowed:
        raise HTTPException(status_code=400, detail=f"Speed must be one of {allowed}")
    with _state_lock:
        _state["speed"] = payload.speed
    return {"speed": payload.speed, "status": "ok"}


@app.get("/speed")
def get_speed():
    with _state_lock:
        return {"speed": _state["speed"]}


# ─────────────────────────────────────────────────────────────────────────────
# QUIZ
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/quiz")
def broadcast_quiz(payload: QuizPayload):
    """
    Teacher sends a quiz.
    Students poll GET /quiz to pick it up.
    """
    if len(payload.options) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 options required")
    quiz = {
        "question":      payload.question,
        "options":       payload.options,
        "correct_index": payload.correct_index,
        "target":        payload.target,
        "sent_at":       time.time(),
    }
    with _state_lock:
        _state["current_quiz"] = quiz
    return {"status": "quiz_sent", "quiz": quiz}


@app.get("/quiz")
def get_quiz():
    """Students poll this to check if a quiz is waiting."""
    with _state_lock:
        return {
            "quiz_pending": _state["current_quiz"] is not None,
            "quiz":         _state["current_quiz"],
        }


@app.post("/quiz/answer")
def submit_answer(payload: QuizAnswerPayload):
    """Student submits quiz answer. Returns correct/incorrect + clears quiz."""
    with _state_lock:
        quiz = _state["current_quiz"]
        if quiz is None:
            raise HTTPException(status_code=404, detail="No active quiz")
        correct = payload.answer_index == quiz["correct_index"]
        if correct:
            _state["current_quiz"] = None   # clear quiz on correct answer

    return {
        "correct":  correct,
        "points":   10 if correct else 0,
        "message":  "Correct! +10 points" if correct else "Try again",
    }


@app.delete("/quiz")
def clear_quiz():
    """Teacher manually clears the current quiz."""
    with _state_lock:
        _state["current_quiz"] = None
    return {"status": "quiz_cleared"}


# ─────────────────────────────────────────────────────────────────────────────
# SESSION CONTROL
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/session/start")
def start_session():
    with _state_lock:
        _state["session_active"] = True
        _live_engine.reset()
    return {"status": "session_started"}


@app.post("/session/end")
def end_session():
    with _state_lock:
        _state["session_active"] = False
    return {"status": "session_ended"}