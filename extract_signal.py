# This file is responsible for extracting finger tapping signals from videos
# using a trained YOLO model and saving them as CSV and graph images.

from ultralytics import YOLO
import cv2
import csv
import math
import os
import matplotlib.pyplot as plt

# -------- SETTINGS --------
VIDEO_DIR = "./PDAV"
MODEL_PATH = "./runs/finger_detector/weights/best.pt"
OUTPUT_DIR = "./signal_output"

CONF_THRES = 0.40
FRAME_SKIP = 2
MAX_FRAMES = 300
MAX_VALID_DISTANCE = 800

TOP_REGION_IGNORE = 0.30
MAX_BOX_WIDTH = 150
MAX_BOX_HEIGHT = 150
# -------------------------

# Create output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load trained YOLO model
model = YOLO(MODEL_PATH)

# Get all video files
video_files = [f for f in os.listdir(VIDEO_DIR) if f.upper().endswith(".MOV")]

for video_file in video_files:
    video_path = os.path.join(VIDEO_DIR, video_file)
    video_name = os.path.splitext(video_file)[0]

    print(f"\nProcessing: {video_name}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        continue

    # Prepare output paths
    csv_path = os.path.join(OUTPUT_DIR, f"detections_{video_name}.csv")
    signal_path = os.path.join(OUTPUT_DIR, f"signal_{video_name}.csv")
    plot_path = os.path.join(OUTPUT_DIR, f"signal_{video_name}.png")

    # -------- STEP 1: DETECTION --------
    # Save raw detections into CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "class", "x", "y", "conf"])

        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SKIP != 0:
                frame_idx += 1
                continue

            h, w, _ = frame.shape

            # Run YOLO detection
            results = model.predict(frame, conf=CONF_THRES, verbose=False)

            for r in results:
                if r.boxes is None:
                    continue

                for b in r.boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    x1, y1, x2, y2 = b.xyxy[0].tolist()

                    # Compute center
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2

                    box_w = x2 - x1
                    box_h = y2 - y1

                    class_name = model.names[cls_id]

                    # -------- FILTERING --------
                    # Ignore face region (top of frame)
                    if y_center < h * TOP_REGION_IGNORE:
                        continue

                    # Ignore large boxes (not fingers)
                    if box_w > MAX_BOX_WIDTH or box_h > MAX_BOX_HEIGHT:
                        continue
                    # --------------------------

                    writer.writerow([frame_idx, class_name, x_center, y_center, conf])

            processed += 1
            if processed >= MAX_FRAMES:
                break

            frame_idx += 1

    cap.release()
    print("Detection done")

    # -------- STEP 2: BUILD SIGNAL --------
    # Read detections and compute distance
    detections = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            cls = row["class"]
            x = float(row["x"])
            y = float(row["y"])
            conf = float(row["conf"])

            if frame not in detections:
                detections[frame] = {}

            # Keep best detection per class
            if cls not in detections[frame] or conf > detections[frame][cls]["conf"]:
                detections[frame][cls] = {"x": x, "y": y, "conf": conf}

    signal = []

    # Compute distance between thumb and index finger
    for frame in sorted(detections.keys()):
        if "thumb" in detections[frame] and "index_finger" in detections[frame]:
            t = detections[frame]["thumb"]
            i = detections[frame]["index_finger"]

            dist = math.sqrt((t["x"] - i["x"])**2 + (t["y"] - i["y"])**2)

            if dist < MAX_VALID_DISTANCE:
                signal.append([frame, dist])

    # Save signal
    with open(signal_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "distance"])
        writer.writerows(signal)

    print("Signal saved")

    # -------- STEP 3: PLOT --------
    frames = [x[0] for x in signal]
    distances = [x[1] for x in signal]

    plt.figure(figsize=(10, 4))
    plt.plot(frames, distances)
    plt.title(video_name)
    plt.xlabel("Frame")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print("Graph saved")

print("\nAll done")