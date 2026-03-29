"""
attentix — video_processor.py
Processes a video file frame-by-frame and returns a full engagement report.
Handles both single student upload and batch (up to 10 videos).
"""

import cv2
import time
from face_engine import FaceEngine

SAMPLE_FPS = 5          # analyse 5 frames per second of video
SEGMENT_SEC = 2.0       # score averaged over 2-second windows


def process_video(video_path: str, student_name: str = "Student") -> dict:
    """
    Process a single video file.
    Returns a full engagement report dict.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Could not open video: {video_path}"}

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps

    # How many frames to skip to hit SAMPLE_FPS
    frame_skip = max(1, int(video_fps / SAMPLE_FPS))

    engine = FaceEngine()
    timeline   = []     # [{time, score, state, signals}]
    all_scores = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every Nth frame
        if frame_idx % frame_skip == 0:
            timestamp = frame_idx / video_fps
            result    = engine.analyse_frame(frame, timestamp)

            all_scores.append(result["score"])
            timeline.append({
                "time":  result["timestamp"],
                "score": result["score"],
                "state": result["state"],
                "signals": result["signals"],
            })

        frame_idx += 1

    cap.release()

    if not all_scores:
        return {"error": "No frames could be analysed"}

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    avg_score = round(sum(all_scores) / len(all_scores))
    max_score = max(all_scores)
    min_score = min(all_scores)

    high_count   = sum(1 for s in all_scores if s >= 70)
    medium_count = sum(1 for s in all_scores if 40 <= s < 70)
    low_count    = sum(1 for s in all_scores if s < 40)
    total        = len(all_scores)

    high_pct   = round((high_count   / total) * 100)
    medium_pct = round((medium_count / total) * 100)
    low_pct    = round((low_count    / total) * 100)

    if avg_score >= 70:
        overall_state = "high"
    elif avg_score >= 40:
        overall_state = "medium"
    else:
        overall_state = "low"

    # ── Per-signal averages ───────────────────────────────────────────────────
    eye_vals    = [t["signals"]["eye_contact_pct"] for t in timeline]
    motion_vals = [t["signals"]["motion"]          for t in timeline]
    inact_vals  = [t["signals"]["inactivity_sec"]  for t in timeline]

    avg_eye_contact  = round(sum(eye_vals)    / len(eye_vals), 1)
    avg_motion       = round(sum(motion_vals) / len(motion_vals), 4)
    max_inactivity   = round(max(inact_vals), 1)

    # Expression mode
    expressions = [t["signals"]["expression"] for t in timeline]
    expr_mode   = max(set(expressions), key=expressions.count)

    # Yawn count (unique events)
    yawn_events = [e for e in engine.events if e["type"] == "yawning"]
    inact_events = [e for e in engine.events if e["type"] == "inactivity"]

    # ── AI feedback ───────────────────────────────────────────────────────────
    feedback = _generate_feedback(
        avg_score, avg_eye_contact, len(yawn_events),
        max_inactivity, expr_mode, low_pct
    )

    return {
        "student":         student_name,
        "duration_sec":    round(duration_sec, 1),
        "frames_analysed": len(timeline),

        "summary": {
            "avg_score":    avg_score,
            "max_score":    max_score,
            "min_score":    min_score,
            "state":        overall_state,
            "high_pct":     high_pct,
            "medium_pct":   medium_pct,
            "low_pct":      low_pct,
        },

        "signals": {
            "avg_eye_contact_pct": avg_eye_contact,
            "avg_motion":          avg_motion,
            "max_inactivity_sec":  max_inactivity,
            "dominant_expression": expr_mode,
            "yawn_count":          len(yawn_events),
        },

        "events": engine.events,

        "timeline": timeline,   # [{time, score, state, signals}]

        "feedback": feedback,
    }


def process_batch(video_paths: list[tuple[str, str]]) -> dict:
    """
    Process up to 10 videos in sequence.
    video_paths = [(path, student_name), ...]
    Returns class-wide report + per-student results.
    """
    results = []
    for path, name in video_paths[:10]:
        r = process_video(path, name)
        results.append(r)

    # Filter out errors
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "No videos could be processed"}

    scores      = [r["summary"]["avg_score"] for r in valid]
    class_avg   = round(sum(scores) / len(scores))
    high_count  = sum(1 for s in scores if s >= 70)
    med_count   = sum(1 for s in scores if 40 <= s < 70)
    low_count   = sum(1 for s in scores if s < 40)
    total_yawns = sum(r["signals"]["yawn_count"] for r in valid)
    total_inact = sum(
        1 for r in valid
        for e in r["events"] if e["type"] == "inactivity"
    )

    # Sort by score descending
    valid_sorted = sorted(valid, key=lambda r: r["summary"]["avg_score"], reverse=True)

    return {
        "class_summary": {
            "student_count":   len(valid),
            "avg_score":       class_avg,
            "high_count":      high_count,
            "medium_count":    med_count,
            "low_count":       low_count,
            "total_yawns":     total_yawns,
            "inactivity_events": total_inact,
        },
        "students": valid_sorted,
        "teacher_alerts": _build_teacher_alerts(valid_sorted),
        "ai_suggestions": _class_suggestions(class_avg, low_count, total_yawns),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _generate_feedback(avg_score, eye_pct, yawn_count, max_inact, expr, low_pct):
    tips = []

    # Attention pattern
    if low_pct > 30:
        tips.append({
            "label": "attention pattern",
            "text": f"Your score dropped below 40 for {low_pct}% of the session. "
                    "Try shorter study blocks (25 min) with 5-min breaks to stay sharper.",
        })
    else:
        tips.append({
            "label": "attention pattern",
            "text": f"Good overall — your score stayed mostly above 40. "
                    f"Avg score was {avg_score}. Try to push it above 70 consistently.",
        })

    # Yawning insight
    if yawn_count >= 2:
        tips.append({
            "label": "yawning insight",
            "text": f"{yawn_count} yawns detected. This often signals fatigue or low oxygen. "
                    "Consider studying after a 10-min walk or at a different time of day.",
        })
    elif yawn_count == 1:
        tips.append({
            "label": "yawning insight",
            "text": "1 yawn detected — mild fatigue. Stay hydrated and keep sessions under 45 minutes.",
        })
    else:
        tips.append({
            "label": "yawning insight",
            "text": "No yawning detected — great alertness throughout the session!",
        })

    # Eye contact / head pose
    if eye_pct >= 70:
        tips.append({
            "label": "what went well",
            "text": f"Eye contact was strong at {eye_pct}% — you were consistently facing the screen. "
                    "Keep it up.",
        })
    else:
        tips.append({
            "label": "what went well",
            "text": f"Eye contact was {eye_pct}% — try to keep your screen at eye level "
                    "to reduce the temptation to look away.",
        })

    return tips


def _build_teacher_alerts(students):
    alerts = []
    for s in students:
        score = s["summary"]["avg_score"]
        name  = s["student"]
        if score < 40:
            alerts.append({
                "type":    "low_engagement",
                "student": name,
                "score":   score,
                "message": f"{name} — score {score}, low engagement throughout session",
            })
        if s["signals"]["yawn_count"] >= 2:
            alerts.append({
                "type":    "yawning",
                "student": name,
                "count":   s["signals"]["yawn_count"],
                "message": f"{name} — {s['signals']['yawn_count']} yawns detected",
            })
        if s["signals"]["max_inactivity_sec"] > 20:
            alerts.append({
                "type":    "inactivity",
                "student": name,
                "seconds": s["signals"]["max_inactivity_sec"],
                "message": f"{name} — inactive for {s['signals']['max_inactivity_sec']}s",
            })
    return alerts


def _class_suggestions(avg, low_count, total_yawns):
    suggestions = []

    if low_count >= 2:
        suggestions.append({
            "label": "pacing alert",
            "text": f"{low_count} students showed low engagement. "
                    "Open your next session with an interactive question before diving into content.",
        })

    if total_yawns >= 3:
        suggestions.append({
            "label": "content insight",
            "text": f"{total_yawns} yawns detected across the class. "
                    "Consider adding a visual analogy or short break at the halfway point.",
        })

    if avg < 60:
        suggestions.append({
            "label": "speed recommendation",
            "text": f"Class average is {avg} — hold playback at 1.0x. "
                    "Only increase to 1.5x when the class average exceeds 70.",
        })
    else:
        suggestions.append({
            "label": "speed recommendation",
            "text": f"Class average is {avg} — good engagement. "
                    "You can push to 1.5x for the next segment.",
        })

    return suggestions