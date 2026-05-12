import cv2
import os

# Change this to your folder path
video_folder = r".\PDAV"
output_base = r"C:.\frames"

os.makedirs(output_base, exist_ok=True)

videos = [f for f in os.listdir(video_folder) if f.endswith(".MOV")]

for video_name in videos:
    video_path = os.path.join(video_folder, video_name)
    video_output = os.path.join(output_base, video_name.split('.')[0])
    os.makedirs(video_output, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {video_name}, total frames: {total_frames}")

    frame_indices = [int(i * total_frames / 30) for i in range(30)]

    count = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(video_output, f"{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            count += 1

    cap.release()

print("DONE extracting frames")