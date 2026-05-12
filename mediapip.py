# This file is responsible for:
# 1. Running live or saved-video finger tracking using MediaPipe
# 2. Detecting thumb tip and index fingertip
# 3. Detecting whether the visible hand is Left or Right
# 4. Computing distance over time
# 5. Counting taps and estimating a simple Parkinson's-style score
# 6. Saving results automatically after each run:
#    - one full signal CSV per run
#    - one summary CSV row per run

import cv2
import math
import csv
import os
import time
from datetime import datetime
from collections import deque, Counter

import numpy as np
import mediapipe as mp

# ---------------- SETTINGS ----------------
# Use 0 for webcam
VIDEO_SOURCE = 0

# Example for saved video:
# VIDEO_SOURCE = r"D:\PDA\PDAV\C30.MOV"

WINDOW_NAME = "MediaPipe Finger Tapping Demo"

SMOOTH_WINDOW = 5
ANALYSIS_WINDOW = 120
TAP_THRESHOLD = 120

OUTPUT_DIR = "live_results"

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# ------------------------------------------


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def estimate_score(signal_values: list[float]) -> int | None:
    """
    Prototype score for demonstration only.
    """
    if len(signal_values) < 20:
        return None

    x = np.array(signal_values, dtype=float)

    # Here amplitude is treated as peak-to-peak for score estimation
    amplitude_for_score = float(np.max(x) - np.min(x))
    std_dev = float(np.std(x))
    speed = float(np.mean(np.abs(np.diff(x)))) if len(x) > 1 else 0.0

    score = 0
    if amplitude_for_score < 200:
        score += 1
    if std_dev > 80:
        score += 1
    if speed < 12:
        score += 1

    return min(4, score)


def save_signal_csv(filepath: str, signal_rows: list[dict]) -> None:
    fieldnames = [
        "frame_index",
        "hand_label",
        "hand_confidence",
        "thumb_x",
        "thumb_y",
        "index_x",
        "index_y",
        "raw_distance",
        "smoothed_distance",
        "tap_count_so_far",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(signal_rows)


def append_summary_csv(filepath: str, summary_row: dict) -> None:
    file_exists = os.path.exists(filepath)

    fieldnames = [
        "session_id",
        "source_type",
        "video_source",
        "total_frames_processed",
        "frames_with_hand_detected",
        "frames_without_hand",
        "session_duration_seconds",
        "tap_count",
        "tap_frequency_hz",
        "taps_per_minute",
        "mean_distance",
        "std_distance",
        "min_distance",
        "max_distance",
        "range_distance",
        "amplitude",
        "mean_speed",
        "estimated_score",
        "dominant_hand_label",
        "left_frames",
        "right_frames",
        "unknown_hand_frames",
        "output_signal_csv",
    ]

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(summary_row)


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    session_id = make_session_id()
    signal_csv_path = os.path.join(OUTPUT_DIR, f"signal_{session_id}.csv")
    summary_csv_path = os.path.join(OUTPUT_DIR, "live_summary_results.csv")

    source_type = "webcam" if VIDEO_SOURCE == 0 else "saved_video"

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    THUMB_TIP = 4
    INDEX_TIP = 8

    distance_history = deque(maxlen=SMOOTH_WINDOW)
    signal_window = deque(maxlen=ANALYSIS_WINDOW)

    full_signal_values: list[float] = []
    signal_rows: list[dict] = []

    tap_count = 0
    prev_state = "OPEN"

    total_frames_processed = 0
    frames_with_hand_detected = 0

    hand_counter = Counter()

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1000, 700)

    print(f"Session started: {session_id}")
    print(f"Source type: {source_type}")
    print(f"Video source: {VIDEO_SOURCE}")

    session_start_time = time.time()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:

        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Finished video or failed to read frame.")
                break

            total_frames_processed += 1
            frame_index += 1

            frame = cv2.resize(frame, (1000, 700))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                frames_with_hand_detected += 1

                hand_landmarks = results.multi_hand_landmarks[0]

                # Handedness detection
                hand_label = "Unknown"
                hand_confidence = 0.0

                if results.multi_handedness:
                    handedness = results.multi_handedness[0]
                    hand_label = handedness.classification[0].label
                    hand_confidence = float(handedness.classification[0].score)

                hand_counter[hand_label] += 1

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = frame.shape

                thumb = hand_landmarks.landmark[THUMB_TIP]
                indexf = hand_landmarks.landmark[INDEX_TIP]

                tx, ty = int(thumb.x * w), int(thumb.y * h)
                ix, iy = int(indexf.x * w), int(indexf.y * h)

                cv2.circle(frame, (tx, ty), 8, (0, 255, 0), -1)
                cv2.circle(frame, (ix, iy), 8, (0, 0, 255), -1)
                cv2.line(frame, (tx, ty), (ix, iy), (255, 0, 0), 2)

                raw_distance = math.sqrt((tx - ix) ** 2 + (ty - iy) ** 2)

                distance_history.append(raw_distance)
                smoothed_distance = float(np.mean(distance_history))

                signal_window.append(smoothed_distance)
                full_signal_values.append(smoothed_distance)

                state = "CLOSE" if smoothed_distance < TAP_THRESHOLD else "OPEN"
                if prev_state == "CLOSE" and state == "OPEN":
                    tap_count += 1
                prev_state = state

                signal_rows.append({
                    "frame_index": frame_index,
                    "hand_label": hand_label,
                    "hand_confidence": round(hand_confidence, 4),
                    "thumb_x": tx,
                    "thumb_y": ty,
                    "index_x": ix,
                    "index_y": iy,
                    "raw_distance": round(raw_distance, 4),
                    "smoothed_distance": round(smoothed_distance, 4),
                    "tap_count_so_far": tap_count,
                })

                score = estimate_score(list(signal_window))

                cv2.putText(
                    frame,
                    f"Hand: {hand_label} ({hand_confidence:.2f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Distance: {smoothed_distance:.1f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Taps: {tap_count}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                )

                if score is not None:
                    cv2.putText(
                        frame,
                        f"Estimated Score: {score}",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                cv2.putText(
                    frame,
                    "Thumb tip (green), Index tip (red)",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (200, 255, 200),
                    2,
                )

            else:
                hand_counter["Unknown"] += 1
                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 200, 200),
                2,
            )

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("User stopped the session.")
                break

    cap.release()
    cv2.destroyAllWindows()

    session_end_time = time.time()
    session_duration_seconds = float(session_end_time - session_start_time)

    save_signal_csv(signal_csv_path, signal_rows)

    if full_signal_values:
        signal_array = np.array(full_signal_values, dtype=float)

        mean_distance = float(np.mean(signal_array))
        std_distance = float(np.std(signal_array))
        min_distance = float(np.min(signal_array))
        max_distance = float(np.max(signal_array))

        # Range = peak-to-peak
        range_distance = float(max_distance - min_distance)

        # Amplitude = half of peak-to-peak range
        amplitude = float(range_distance / 2.0)

        mean_speed = float(np.mean(np.abs(np.diff(signal_array)))) if len(signal_array) > 1 else 0.0
        estimated_score = estimate_score(full_signal_values)
    else:
        mean_distance = 0.0
        std_distance = 0.0
        min_distance = 0.0
        max_distance = 0.0
        range_distance = 0.0
        amplitude = 0.0
        mean_speed = 0.0
        estimated_score = None

    frames_without_hand = total_frames_processed - frames_with_hand_detected

    dominant_hand_label = "Unknown"
    if hand_counter:
        dominant_hand_label = hand_counter.most_common(1)[0][0]

    tap_frequency_hz = float(tap_count / session_duration_seconds) if session_duration_seconds > 0 else 0.0
    taps_per_minute = float(tap_frequency_hz * 60.0)

    summary_row = {
        "session_id": session_id,
        "source_type": source_type,
        "video_source": VIDEO_SOURCE,
        "total_frames_processed": total_frames_processed,
        "frames_with_hand_detected": frames_with_hand_detected,
        "frames_without_hand": frames_without_hand,
        "session_duration_seconds": round(session_duration_seconds, 4),
        "tap_count": tap_count,
        "tap_frequency_hz": round(tap_frequency_hz, 4),
        "taps_per_minute": round(taps_per_minute, 4),
        "mean_distance": round(mean_distance, 4),
        "std_distance": round(std_distance, 4),
        "min_distance": round(min_distance, 4),
        "max_distance": round(max_distance, 4),
        "range_distance": round(range_distance, 4),
        "amplitude": round(amplitude, 4),
        "mean_speed": round(mean_speed, 4),
        "estimated_score": estimated_score,
        "dominant_hand_label": dominant_hand_label,
        "left_frames": hand_counter.get("Left", 0),
        "right_frames": hand_counter.get("Right", 0),
        "unknown_hand_frames": hand_counter.get("Unknown", 0),
        "output_signal_csv": signal_csv_path,
    }

    append_summary_csv(summary_csv_path, summary_row)

    print("\nSession finished.")
    print(f"Signal CSV saved to: {signal_csv_path}")
    print(f"Summary CSV updated at: {summary_csv_path}")
    print("Summary row:")
    print(summary_row)


if __name__ == "__main__":
    main()