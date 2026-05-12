from ultralytics import YOLO
import cv2
import math
import numpy as np
from collections import deque

MODEL_PATH = "./runs/finger_detector/weights/best.pt"
VIDEO_SOURCE = "./PDAV/C1_LEFT.MOV" # 0 = webcam, or "./PDAV/PD1_RIGHT.MOV"
CONF_THRES = 0.25
MAX_VALID_DISTANCE = 800
SMOOTH_WINDOW = 5
ANALYSIS_WINDOW = 120   # number of recent frames used for scoring

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam/video source")

distance_history = deque(maxlen=SMOOTH_WINDOW)
signal_window = deque(maxlen=ANALYSIS_WINDOW)

tap_count = 0
prev_state = "OPEN"

def get_best_points(results, model_names):
    best = {}
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            class_name = model_names[cls_id]

            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            if class_name not in best or conf > best[class_name]["conf"]:
                best[class_name] = {
                    "x": x_center,
                    "y": y_center,
                    "conf": conf,
                    "box": (int(x1), int(y1), int(x2), int(y2))
                }
    return best

def estimate_updrs_from_signal(signal):
    """
    Heuristic UPDRS-style estimator based on:
    - amplitude (range)
    - variability (std)
    - speed (mean abs diff)
    - tap count (local minima count)
    """
    if len(signal) < 20:
        return None, "Insufficient data", {}

    x = np.array(signal, dtype=float)

    mean_dist = float(np.mean(x))
    std_dist = float(np.std(x))
    min_dist = float(np.min(x))
    max_dist = float(np.max(x))
    amplitude = max_dist - min_dist
    mean_speed = float(np.mean(np.abs(np.diff(x)))) if len(x) > 1 else 0.0

    # local minima as tap count
    taps = 0
    for i in range(1, len(x) - 1):
        if x[i] < x[i - 1] and x[i] < x[i + 1]:
            taps += 1

    # Heuristic scoring rules
    score = 0

    # Reduced amplitude
    if amplitude < 180:
        score += 2
    elif amplitude < 260:
        score += 1

    # Irregularity / variability
    if std_dist > 90:
        score += 1
    if std_dist > 130:
        score += 1

    # Reduced speed
    if mean_speed < 10:
        score += 1
    elif mean_speed < 16:
        score += 0.5

    # Reduced tap frequency
    if taps < 4:
        score += 1
    elif taps < 7:
        score += 0.5

    # Clamp to 0–4
    score = int(round(min(4, max(0, score))))

    severity_map = {
        0: "Normal / No clear impairment",
        1: "Slight impairment",
        2: "Mild impairment",
        3: "Moderate impairment",
        4: "Severe impairment"
    }

    features = {
        "mean_distance": round(mean_dist, 1),
        "std_distance": round(std_dist, 1),
        "amplitude": round(amplitude, 1),
        "mean_speed": round(mean_speed, 1),
        "tap_count": taps
    }

    return score, severity_map[score], features

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (900, 600))
    results = model.predict(frame, conf=CONF_THRES, verbose=False)
    best = get_best_points(results, model.names)

    for class_name, info in best.items():
        x1, y1, x2, y2 = info["box"]
        label = f"{class_name} {info['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (info["x"], info["y"]), 4, (0, 0, 255), -1)
        cv2.putText(
            frame, label, (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
        )

    score_text = "Estimating..."
    severity_text = ""
    feature_texts = []

    if "index_finger" in best and "thumb" in best:
        idx = best["index_finger"]
        th = best["thumb"]

        cv2.line(frame, (idx["x"], idx["y"]), (th["x"], th["y"]), (255, 0, 0), 2)

        dist = math.sqrt((idx["x"] - th["x"])**2 + (idx["y"] - th["y"])**2)

        if dist < MAX_VALID_DISTANCE:
            distance_history.append(dist)
            smooth_distance = float(np.mean(distance_history))
            signal_window.append(smooth_distance)

            # simple open/close tap counting
            state = "CLOSE" if smooth_distance < 120 else "OPEN"
            if prev_state == "CLOSE" and state == "OPEN":
                tap_count += 1
            prev_state = state

            est_score, severity, features = estimate_updrs_from_signal(signal_window)

            cv2.putText(
                frame, f"Distance: {smooth_distance:.1f}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
            cv2.putText(
                frame, f"Tap Count: {tap_count}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2
            )

            if est_score is not None:
                score_text = f"Estimated UPDRS: {est_score}"
                severity_text = severity
                feature_texts = [
                    f"Amp: {features['amplitude']}",
                    f"Std: {features['std_distance']}",
                    f"Speed: {features['mean_speed']}",
                    f"Taps(win): {features['tap_count']}"
                ]
        else:
            cv2.putText(
                frame, "Distance spike ignored",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
    else:
        cv2.putText(
            frame, "Thumb/index pair not detected",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

    cv2.putText(
        frame, score_text,
        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    cv2.putText(
        frame, severity_text,
        (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2
    )

    y0 = 170
    for i, txt in enumerate(feature_texts):
        cv2.putText(
            frame, txt,
            (20, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2
        )

    cv2.putText(
        frame,
        "Prototype only - not a clinical diagnosis",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2
    )

    cv2.imshow("Real-Time Parkinson Estimator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()