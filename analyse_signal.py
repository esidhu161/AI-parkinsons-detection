import csv
import numpy as np
import os

SIGNAL_DIR = "./signal_output"
OUTPUT_FEATURES = "./signal_output/features.csv"

# Optional: add UPDRS scores manually here for report use
# If unknown, leave as None
UPDRS_MAP = {
    "PD1_RIGHT": 3,
    "PD1_LEFT": 2,
    "PD2_RIGHT": 2,
    "PD2_LEFT": 3,
    "PD3_RIGHT": 4,
    "PD3_LEFT": 3,
    "C1_RIGHT": 0,
    "C1_LEFT": 0,
    "C2_RIGHT": 0,
    "C2_LEFT": 0,
}

signal_files = [
    f for f in os.listdir(SIGNAL_DIR)
    if f.startswith("signal_") and f.endswith(".csv") and f != "features.csv"
]

if not signal_files:
    print("No signal files found in ./signal_output")
    exit()

with open(OUTPUT_FEATURES, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "video_name",
        "group",
        "hand",
        "mean_distance",
        "std_distance",
        "min_distance",
        "max_distance",
        "num_taps",
        "mean_speed",
        "updrs_score"
    ])

    for signal_file in signal_files:
        signal_path = os.path.join(SIGNAL_DIR, signal_file)
        video_name = signal_file.replace("signal_", "").replace(".csv", "")

        frames = []
        distances = []

        with open(signal_path, "r") as sf:
            reader = csv.DictReader(sf)
            for row in reader:
                frames.append(int(row["frame"]))
                distances.append(float(row["distance"]))

        if len(distances) < 3:
            print(f"Skipping {video_name}: not enough signal data")
            continue

        distances = np.array(distances)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # tap detection using local minima
        peaks = []
        for i in range(1, len(distances) - 1):
            if distances[i] < distances[i - 1] and distances[i] < distances[i + 1]:
                peaks.append(i)

        num_taps = len(peaks)

        velocity = np.diff(distances)
        mean_speed = np.mean(np.abs(velocity)) if len(velocity) > 0 else 0.0

        # infer group and hand from filename
        upper_name = video_name.upper()
        group = "PD" if upper_name.startswith("PD") else "Control"
        hand = "RIGHT" if "RIGHT" in upper_name else "LEFT"

        updrs_score = UPDRS_MAP.get(video_name, None)

        writer.writerow([
            video_name,
            group,
            hand,
            round(mean_dist, 2),
            round(std_dist, 2),
            round(min_dist, 2),
            round(max_dist, 2),
            num_taps,
            round(mean_speed, 2),
            updrs_score
        ])

        print(f"Processed features for: {video_name}")

print(f"\nFeatures saved to: {OUTPUT_FEATURES}")