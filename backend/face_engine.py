"""
attentix — face_engine.py
Fixed: fresh state per video, lower yawn threshold, faster yawn detection.
"""

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
UPPER_LIP = 13
LOWER_LIP = 14
NOSE_TIP  = 1
CHIN      = 152
LEFT_EYE_CORNER  = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH  = 61
RIGHT_MOUTH = 291

MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,   -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0,  -150.0, -125.0),
], dtype=np.float64)


def _ear(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return float((A + B) / (2.0 * C)) if C > 0 else 0.0


def _head_pose(landmarks, w, h):
    pts = np.array([
        (landmarks[NOSE_TIP].x * w,        landmarks[NOSE_TIP].y * h),
        (landmarks[CHIN].x * w,             landmarks[CHIN].y * h),
        (landmarks[LEFT_EYE_CORNER].x * w,  landmarks[LEFT_EYE_CORNER].y * h),
        (landmarks[RIGHT_EYE_CORNER].x * w, landmarks[RIGHT_EYE_CORNER].y * h),
        (landmarks[LEFT_MOUTH].x * w,       landmarks[LEFT_MOUTH].y * h),
        (landmarks[RIGHT_MOUTH].x * w,      landmarks[RIGHT_MOUTH].y * h),
    ], dtype=np.float64)
    cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, pts, cam, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return float(angles[0]), float(angles[1]), float(angles[2])


def _lip_ratio(landmarks, h):
    upper = landmarks[UPPER_LIP].y * h
    lower = landmarks[LOWER_LIP].y * h
    fh = abs(landmarks[152].y * h - landmarks[10].y * h)
    return float(abs(lower - upper) / fh) if fh > 0 else 0.0


def _motion(prev, curr):
    if prev is None:
        return 0.05
    return float(np.mean([
        np.sqrt((c.x - p.x)**2 + (c.y - p.y)**2)
        for c, p in zip(curr, prev)
    ]))


def _expression(ear, lip_r, yaw):
    if lip_r > 0.07:   return "drowsy"
    if abs(yaw) > 45:  return "distracted"
    if ear < 0.15:     return "drowsy"
    if ear > 0.24:     return "focused"
    return "neutral"


def _calc_score(ear, yaw, pitch, motion, lip_r, inact, expr):
    s = 0
    if ear > 0.15:                                  s += 40
    if abs(yaw) < 45 and abs(pitch) < 45:           s += 30
    if motion > 0.0003:                             s += 20
    if expr in ("focused", "neutral", "attentive"): s += 10
    if abs(yaw) > 55:                               s -= 30
    if lip_r > 0.07:                                s -= 15
    if expr == "drowsy":                            s -= 10
    if inact > 30:                                  s -= 10
    return max(0, min(100, s))


class FaceEngine:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._reset_state()

    def _reset_state(self):
        """Reset all stateful fields — call this before each new video."""
        self._prev      = None
        self._inact_t   = None
        self._yawn_t    = None
        self.events     = []
        self._buf       = []
        self._last_t    = 0.0
        self._score     = 70

    def analyse_frame(self, frame, ts: float) -> dict:
        h, w = frame.shape[:2]
        res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return self._no_face(ts)

        lm = res.multi_face_landmarks[0].landmark

        # 1. Eye contact
        ear = (_ear(lm, LEFT_EYE, w, h) + _ear(lm, RIGHT_EYE, w, h)) / 2.0
        eyes_open = bool(ear > 0.15)
        eye_pct   = float(round(min(100.0, ear / 0.22 * 100), 1))

        # 2. Head pose
        pitch, yaw, _ = _head_pose(lm, w, h)
        if abs(yaw) < 45 and abs(pitch) < 45:  pose = "forward"
        elif abs(yaw) < 55:                    pose = "slight tilt"
        elif abs(pitch) > 40:                  pose = "down"
        else:                                  pose = "away"

        # 3. Lip / yawn — triggers after 0.8s of open mouth
        lip_r   = _lip_ratio(lm, h)
        yawning = False
        if lip_r > 0.07:
            if self._yawn_t is None:
                self._yawn_t = ts
            elif ts - self._yawn_t > 0.8:
                yawning = True
                already = any(
                    e["type"] == "yawning" and abs(e["time"] - ts) < 3.0
                    for e in self.events
                )
                if not already:
                    self.events.append({
                        "type":       "yawning",
                        "time":       float(round(ts, 2)),
                        "confidence": float(round(min(0.99, 0.75 + lip_r * 2), 2)),
                    })
                self._yawn_t = None
        else:
            self._yawn_t = None

        # 4. Motion + inactivity
        motion = _motion(self._prev, lm)
        self._prev = lm
        inact = 0.0
        if motion < 0.0003:
            if self._inact_t is None:
                self._inact_t = ts
            inact = float(ts - self._inact_t)
            if inact > 30:
                already = any(
                    e["type"] == "inactivity" and
                    e.get("start") == float(round(self._inact_t, 2))
                    for e in self.events
                )
                if not already:
                    self.events.append({
                        "type":       "inactivity",
                        "start":      float(round(self._inact_t, 2)),
                        "end":        float(round(ts, 2)),
                        "confidence": 0.87,
                    })
        else:
            self._inact_t = None

        # 5. Expression
        expr = _expression(ear, lip_r, yaw)

        # 6. Score — averaged every 4 seconds
        raw = _calc_score(ear, yaw, pitch, motion, lip_r, inact, expr)
        self._buf.append(raw)
        if ts - self._last_t >= 4.0:
            self._score  = int(round(sum(self._buf) / len(self._buf)))
            self._buf    = []
            self._last_t = float(ts)

        sc    = int(self._score)
        state = "high" if sc >= 70 else "medium" if sc >= 40 else "low"
        act   = "speed_1.5x" if sc >= 70 else "speed_1.0x" if sc >= 40 else "pause_and_quiz"
        alert = bool(sc < 40 or inact > 30)
        msg   = (f"Inactive {int(inact)}s" if inact > 30
                 else "Low engagement" if sc < 40 else "")

        return {
            "score":      sc,
            "state":      state,
            "confidence": float(round(min(0.99, 0.6 + ear * 0.8), 2)),
            "timestamp":  float(round(ts, 2)),
            "signals": {
                "eye_contact_pct": float(eye_pct),
                "eyes_open":       bool(eyes_open),
                "head_pose":       str(pose),
                "yaw_deg":         float(round(yaw, 1)),
                "pitch_deg":       float(round(pitch, 1)),
                "yawning":         bool(yawning),
                "lip_ratio":       float(round(lip_r, 3)),
                "motion":          float(round(motion, 4)),
                "inactivity_sec":  float(round(inact, 1)),
                "expression":      str(expr),
            },
            "events":  self.events[-10:],
            "action":  str(act),
            "teacher_alert": {
                "triggered": bool(alert),
                "message":   str(msg),
            },
        }

    def _no_face(self, ts):
        return {
            "score":      0,
            "state":      "no_face",
            "confidence": 0.0,
            "timestamp":  float(round(ts, 2)),
            "signals": {
                "eye_contact_pct": 0.0,
                "eyes_open":       False,
                "head_pose":       "unknown",
                "yaw_deg":         0.0,
                "pitch_deg":       0.0,
                "yawning":         False,
                "lip_ratio":       0.0,
                "motion":          0.0,
                "inactivity_sec":  0.0,
                "expression":      "unknown",
            },
            "events":  [],
            "action":  "pause_and_quiz",
            "teacher_alert": {
                "triggered": True,
                "message":   "No face detected",
            },
        }

    def reset(self):
        self._reset_state()