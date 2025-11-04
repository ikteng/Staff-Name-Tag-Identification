import cv2
import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# --- Settings ---
video_path = "sample.mp4"
output_dir = "to_label"
min_conf = 0.5  # YOLO confidence threshold
min_crop_w = 30  # skip too small crops
min_crop_h = 30
os.makedirs(output_dir, exist_ok=True)

# --- Load Models ---
yolo = YOLO("yolov8m.pt")
pose = YOLO("yolov8n-pose.pt")

# --- Video Setup ---
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[INFO] Processing {frame_count} frames from {video_path}...")

frame_idx = 0
crop_idx = 0
# saved_hashes = set()  # prevent duplicates

# --- Frame Processing ---
for _ in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            if int(box.cls) != 0 or float(box.conf) < min_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            pose_res = pose(person_crop, verbose=False)
            for pr in pose_res:
                if pr.keypoints is None or len(pr.keypoints.xy) == 0:
                    continue
                keypoints = pr.keypoints.xy[0].cpu().numpy()
                conf = pr.keypoints.conf[0].cpu().numpy()

                # Need at least shoulders
                if conf[5] < 0.3 or conf[6] < 0.3:
                    continue

                # Shoulders + hips define upper-body box
                left_shoulder, right_shoulder = keypoints[5], keypoints[6]
                left_hip, right_hip = keypoints[11], keypoints[12]

                x_min = int(min(left_shoulder[0], right_shoulder[0])) + x1
                x_max = int(max(left_shoulder[0], right_shoulder[0])) + x1
                y_min = int(min(left_shoulder[1], right_shoulder[1])) + y1
                y_max = int(max(left_hip[1], right_hip[1])) + y1 if conf[11] > 0.3 or conf[12] > 0.3 else int(y1 + (y2 - y1) * 0.6)

                # Add margin
                margin_x = int((x_max - x_min) * 0.2)
                margin_y = int((y_max - y_min) * 0.25)
                x_min, x_max = max(0, x_min - margin_x), min(width, x_max + margin_x)
                y_min, y_max = max(0, y_min - margin_y), min(height, y_max + margin_y)

                crop = frame[y_min:y_max, x_min:x_max]
                if crop.size == 0:
                    continue

                # Skip tiny boxes
                if crop.shape[1] < min_crop_w or crop.shape[0] < min_crop_h:
                    continue

                # # Skip near-duplicates using hash
                # img_hash = hash(crop.tobytes()[:1000])
                # if img_hash in saved_hashes:
                #     continue
                # saved_hashes.add(img_hash)

                save_path = os.path.join(output_dir, f"crop_{frame_idx:05d}_{crop_idx:03d}.jpg")
                cv2.imwrite(save_path, crop)
                crop_idx += 1

    frame_idx += 1

cap.release()
print(f"\n✅ Done! Saved {crop_idx} upper-body crops to '{output_dir}/'")
