from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from database import (
    get_mode_statistics,
    get_recent_sessions,
    initialize_database,
    save_session,
)


MODEL_PATH = Path("models/pose_landmarker_full.task")

VISIBILITY_THRESHOLD = 0.6
PUSHUP_DOWN_THRESHOLD = 90
PUSHUP_UP_THRESHOLD = 140
SQUAT_DOWN_THRESHOLD = 110
SQUAT_UP_THRESHOLD = 160

SESSION_DURATION = 10
COUNTDOWN_DURATION = 5
TARGET_OPTIONS = (5, 10, 15, 20)

LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15
LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27


def calculate_angle(first, vertex, last) -> float:
    first_angle = np.arctan2(
        first.y - vertex.y,
        first.x - vertex.x,
    )
    last_angle = np.arctan2(
        last.y - vertex.y,
        last.x - vertex.x,
    )
    angle = abs(np.degrees(last_angle - first_angle))
    return min(angle, 360 - angle)


def reliable_angle(landmarks: list, first_index: int, vertex_index: int, last_index: int):
    points = [
        landmarks[first_index],
        landmarks[vertex_index],
        landmarks[last_index],
    ]

    if any(point.visibility <= VISIBILITY_THRESHOLD for point in points):
        return None

    return calculate_angle(*points)


def get_database_mode(game_mode: str, target_repetitions: int) -> str:
    if game_mode == "TARGET":
        return f"TARGET_{target_repetitions}"
    return game_mode


def draw_pose(frame, pose_landmarks: list) -> int:
    height, width = frame.shape[:2]
    reliable_landmarks = [
        landmark
        for landmark in pose_landmarks
        if landmark.visibility > VISIBILITY_THRESHOLD
    ]
    reliable_indexes = {
        index
        for index, landmark in enumerate(pose_landmarks)
        if landmark.visibility > VISIBILITY_THRESHOLD
    }

    for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        if connection.start not in reliable_indexes:
            continue
        if connection.end not in reliable_indexes:
            continue

        start = pose_landmarks[connection.start]
        end = pose_landmarks[connection.end]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame, start_point, end_point, (255, 255, 0), 2)

    for landmark in reliable_landmarks:
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, point, 4, (0, 255, 0), -1)

    return len(reliable_landmarks)


def run() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encuentra el modelo: {MODEL_PATH}. "
            "Ejecuta primero download_model.py."
        )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    initialize_database()

    base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    timestamp_ms = 0
    pushup_state = "UP"
    pushup_count = 0
    squat_state = "UP"
    squat_count = 0
    selected_exercise = "FLEXIONES"
    game_mode = "TIMED"
    game_state = "MENU"
    countdown_start = None
    session_start = None
    final_elapsed = 0
    session_saved = False
    statistics = (0, 0, 0.0, 0.0)
    recent_sessions = []
    target_index = 0
    target_repetitions = TARGET_OPTIONS[target_index]

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("No se pudo leer un fotograma de la cámara.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )
                result = landmarker.detect_for_video(image, timestamp_ms)
                timestamp_ms += 33

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    reliable_count = draw_pose(frame, landmarks)
                    elbow_angle = reliable_angle(
                        landmarks,
                        LEFT_SHOULDER,
                        LEFT_ELBOW,
                        LEFT_WRIST,
                    )
                    knee_angle = reliable_angle(
                        landmarks,
                        LEFT_HIP,
                        LEFT_KNEE,
                        LEFT_ANKLE,
                    )
                else:
                    reliable_count = 0
                    elbow_angle = None
                    knee_angle = None

                now = time.monotonic()

                if game_state == "COUNTDOWN":
                    countdown_remaining = max(
                        0,
                        COUNTDOWN_DURATION - (now - countdown_start),
                    )
                    if countdown_remaining == 0:
                        game_state = "ACTIVE"
                        session_start = now

                if game_state == "ACTIVE":
                    if game_mode == "TIMED":
                        remaining_time = max(
                            0,
                            SESSION_DURATION - (now - session_start),
                        )
                        if remaining_time == 0:
                            final_elapsed = now - session_start
                            game_state = "FINISHED"
                    elif game_mode == "TARGET":
                        elapsed_time = now - session_start

                if reliable_count:
                    message = f"Pose detectada: {reliable_count} puntos fiables"
                    color = (0, 200, 0)
                else:
                    message = "No se detecta una pose fiable"
                    color = (0, 0, 255)
    
                if (
                    game_state == "ACTIVE"
                    and selected_exercise == "FLEXIONES"
                    and elbow_angle is not None
                ):
                    if (
                        pushup_state == "UP"
                        and elbow_angle < PUSHUP_DOWN_THRESHOLD
                    ):
                        pushup_state = "DOWN"
                    elif (
                        pushup_state == "DOWN"
                        and elbow_angle > PUSHUP_UP_THRESHOLD
                    ):
                        pushup_state = "UP"
                        pushup_count += 1

                if (
                    game_state == "ACTIVE"
                    and selected_exercise == "SENTADILLAS"
                    and knee_angle is not None
                ):
                    if (
                        squat_state == "UP"
                        and knee_angle < SQUAT_DOWN_THRESHOLD
                    ):
                        squat_state = "DOWN"
                    elif (
                        squat_state == "DOWN"
                        and knee_angle > SQUAT_UP_THRESHOLD
                    ):
                        squat_state = "UP"
                        squat_count += 1

                cv2.putText(
                    frame,
                    message,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                if elbow_angle is not None:
                    cv2.putText(
                        frame,
                        f"Codo izquierdo: {elbow_angle:.0f} grados",
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (255, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )

                if knee_angle is not None:
                    cv2.putText(
                        frame,
                        f"Rodilla izquierda: {knee_angle:.0f} grados",
                        (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (255, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )

                if selected_exercise == "FLEXIONES":
                    state = pushup_state
                    count = pushup_count
                else:
                    state = squat_state
                    count = squat_count

                mode_labels = {
                    "TIMED": "Reto 10 segundos",
                    "FREE": "Entrenamiento libre",
                    "TARGET": f"Fijar {target_repetitions} repeticiones",
                }

                if (
                    game_state == "ACTIVE"
                    and game_mode == "TARGET"
                    and count >= target_repetitions
                ):
                    final_elapsed = time.monotonic() - session_start
                    game_state = "FINISHED"

                cv2.putText(
                    frame,
                    f"Ejercicio: {selected_exercise}",
                    (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"Estado: {state}  Repeticiones: {count}",
                    (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "1: Flexiones   2: Sentadillas",
                    (20, 255),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"Modo: {mode_labels[game_mode]}",
                    (20, 335),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if game_state == "MENU":
                    cv2.putText(
                        frame,
                        "m: Reto  l: Libre  f: Fijar  s: Estadisticas",
                        (20, 375),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                elif game_state == "COUNTDOWN":
                    cv2.putText(
                        frame,
                        f"Comenzamos en: {countdown_remaining:.0f}",
                        (20, 375),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                elif game_state == "ACTIVE":
                    score = count * 10
                    if game_mode == "TIMED":
                        active_message = (
                            f"Tiempo: {remaining_time:.0f} s  "
                            f"Puntuacion: {score}"
                        )
                    elif game_mode == "FREE":
                        active_message = (
                            f"Entrenamiento libre  "
                            f"Puntuacion: {score}  ENTER: terminar"
                        )
                    else:
                        active_message = (
                            f"Objetivo: {target_repetitions}  "
                            f"Tiempo: {elapsed_time:.2f} s"
                        )
                    cv2.putText(
                        frame,
                        active_message,
                        (20, 375),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                elif game_state == "FINISHED":
                    frame[:] = 0
                    cv2.putText(
                        frame,
                        "SESION TERMINADA",
                        (80, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.4,
                        (0, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        (
                            f"{selected_exercise}: {count}  "
                            f"Puntuacion: {count * 10}"
                        ),
                        (80, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    if game_mode == "TARGET":
                        cv2.putText(
                            frame,
                            f"Tiempo empleado: {final_elapsed:.2f} s",
                            (80, 245),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.6,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    cv2.putText(
                        frame,
                        "Pulsa r para volver al menu o q para salir",
                        (80, 330),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    frame[:] = 0
                    sessions, repetitions, average_duration, average_score = statistics
                    cv2.putText(
                        frame,
                        "ESTADISTICAS",
                        (80, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.4,
                        (0, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Modo: {mode_labels[game_mode]}",
                        (80, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Sesiones: {sessions}",
                        (80, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Repeticiones totales: {repetitions}",
                        (80, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Duracion media: {average_duration:.2f} s",
                        (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Puntuacion media: {average_score:.2f}",
                        (80, 280),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        "Ultimas sesiones:",
                        (80, 320),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    for index, session in enumerate(recent_sessions):
                        date, repetitions, duration, score = session
                        session_text = (
                            f"{date[:16]} | {repetitions} reps | "
                            f"{duration:.1f} s | {score} pts"
                        )
                        cv2.putText(
                            frame,
                            session_text,
                            (80, 350 + index * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    cv2.putText(
                        frame,
                        "Pulsa r para volver al menu",
                        (80, 440),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                if game_state == "FINISHED" and not session_saved:
                    score = count * 10
                    save_session(
                        selected_exercise,
                        get_database_mode(game_mode, target_repetitions),
                        count,
                        final_elapsed,
                        score,
                    )
                    session_saved = True

                cv2.imshow("Pose Landmarker", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r") and game_state in ("FINISHED", "STATS"):
                    pushup_state = "UP"
                    pushup_count = 0
                    squat_state = "UP"
                    squat_count = 0
                    game_state = "MENU"
                elif key == ord("1") and game_state == "MENU":
                    selected_exercise = "FLEXIONES"
                elif key == ord("2") and game_state == "MENU":
                    selected_exercise = "SENTADILLAS"
                elif key == ord("m") and game_state == "MENU":
                    game_mode = "TIMED"
                elif key == ord("l") and game_state == "MENU":
                    game_mode = "FREE"
                elif key == ord("f") and game_state == "MENU":
                    if game_mode != "TARGET":
                        game_mode = "TARGET"
                    else:
                        target_index = (target_index + 1) % len(TARGET_OPTIONS)
                        target_repetitions = TARGET_OPTIONS[target_index]
                elif key == ord("s") and game_state == "MENU":
                    database_mode = get_database_mode(
                        game_mode,
                        target_repetitions,
                    )
                    statistics = get_mode_statistics(database_mode)
                    recent_sessions = get_recent_sessions(database_mode)
                    game_state = "STATS"
                elif key in (13, 10) and game_state == "MENU":
                    countdown_start = time.monotonic()
                    game_state = "COUNTDOWN"
                    session_saved = False
                elif (
                    key in (13, 10)
                    and game_state == "ACTIVE"
                    and game_mode == "FREE"
                ):
                    final_elapsed = time.monotonic() - session_start
                    game_state = "FINISHED"

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
