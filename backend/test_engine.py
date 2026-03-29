"""
attentix — test_engine.py
Visualizes face mesh landmarks + live engagement score.
Run: python test_engine.py
     python test_engine.py video.mp4
"""

import sys
import cv2
import time
import numpy as np
import mediapipe as mp
from face_engine import FaceEngine

mp_face_mesh      = mp.solutions.face_mesh
mp_drawing        = mp.solutions.drawing_utils

LANDMARK_SPEC = mp_drawing.DrawingSpec(
    color=(0, 255, 120), thickness=1, circle_radius=1
)
CONNECTION_SPEC = mp_drawing.DrawingSpec(
    color=(0, 200, 80), thickness=1
)


def draw_overlay(frame, result):
    sig   = result["signals"]
    score = result["score"]
    state = result["state"].upper()

    if score >= 70:   color = (0, 220, 120)
    elif score >= 40: color = (0, 165, 255)
    else:             color = (0, 80, 220)

    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Score: {score}  {state}",
                (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.putText(frame,
                f"Eye: {sig['eye_contact_pct']:.0f}%  Pose: {sig['head_pose']}  Yaw: {sig['yaw_deg']:.0f}deg",
                (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame,
                f"Expr: {sig['expression']}  Inact: {sig['inactivity_sec']:.0f}s  Motion: {sig['motion']:.4f}",
                (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Action: {result['action']}",
                (16, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 220), 1)

    bar_w = int((score / 100) * w)
    cv2.rectangle(frame, (0, h - 8), (w, h), (30, 30, 40), -1)
    cv2.rectangle(frame, (0, h - 8), (bar_w, h), color, -1)


def test_webcam():
    print("Opening webcam — press Q to quit\n")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    engine = FaceEngine()
    start  = time.time()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh_vis:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ts     = time.time() - start
            result = engine.analyse_frame(frame.copy(), ts)

            # Draw face mesh (green dots like image 2)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            mesh_result = face_mesh_vis.process(rgb)
            rgb.flags.writeable = True

            if mesh_result.multi_face_landmarks:
                for face_lm in mesh_result.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_lm,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=LANDMARK_SPEC,
                        connection_drawing_spec=CONNECTION_SPEC,
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_lm,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 255, 180), thickness=1
                        ),
                    )

            draw_overlay(frame, result)

            sig = result["signals"]
            print(
                f"\r[{ts:5.1f}s] score={result['score']:3d} "
                f"state={result['state']:6s} "
                f"eye={sig['eye_contact_pct']:4.0f}% "
                f"pose={sig['head_pose']:11s} "
                f"yaw={sig['yaw_deg']:+5.0f}deg "
                f"expr={sig['expression']:10s} "
                f"inact={sig['inactivity_sec']:3.0f}s",
                end="", flush=True
            )

            cv2.imshow("Attentix — Face Engine Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n\nEvents: {engine.events}")


def test_video(path):
    print(f"Processing: {path}\n")
    from video_processor import process_video
    report = process_video(path, "Test Student")
    if "error" in report:
        print(f"ERROR: {report['error']}")
        return
    s = report["summary"]
    sig = report["signals"]
    print(f"Avg score: {s['avg_score']} ({s['state'].upper()})")
    print(f"High: {s['high_pct']}%  Med: {s['medium_pct']}%  Low: {s['low_pct']}%")
    print(f"Eye: {sig['avg_eye_contact_pct']}%  Yawns: {sig['yawn_count']}  Max inact: {sig['max_inactivity_sec']}s")
    print(f"Events: {report['events']}")
    for fb in report["feedback"]:
        print(f"\n[{fb['label']}] {fb['text']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_video(sys.argv[1])
    else:
        test_webcam()